import json
import pickle
import hydra
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from pathlib import Path
from flax import nnx

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run, is_converged
from projectlib.models import MLP
from projectlib.utils import noop

ACTIVATION_FNS = {
    "identity": noop,
    "noop": noop,
    "tanh": nnx.tanh,
    "relu": nnx.relu,
}

def concatenate_histories(old: list[jnp.ndarray] | None, new: list[jnp.ndarray]) -> list[jnp.ndarray]:
    if old is None:
        return new
    else:
        return [jnp.concatenate([old_i, new_i[1:]], axis=0) # drop overlapping timestep
                for old_i, new_i in zip(old, new)]

def to_numpy(x):
    if isinstance(x, (tuple, list)):
        return type(x)(to_numpy(xi) for xi in x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, jnp.ndarray):
        return np.asarray(x)
    else:
        return x

def run_single_linearization(model, x: jnp.ndarray, rclamp: jnp.ndarray, cfg, timesteps: int):
    """Run approx-PC for a fixed number of steps with one linearization."""
    vjps = current_pc_vjp(model, x)
    _, history, t = run(model, approx_pc_step, x,
                        vjps=vjps,
                        k=cfg.approx_pc.k,
                        max_timesteps=timesteps,
                        tolerance=cfg.dynamics.convergence_tolerance,
                        tau=cfg.dynamics.time_constant,
                        learn=False,
                        rclamp=rclamp,
                        return_act_history=True)
    converged = is_converged(history, cfg.dynamics.convergence_tolerance, "norm")

    return history, t, converged

def run_with_relinearize(model, x: jnp.ndarray, rclamp: jnp.ndarray, cfg,
                         total_timesteps: int | None = None) -> tuple[list[jnp.ndarray], int]:
    """Run approx-PC while optionally re-linearizing every fixed number of steps.

    Returns (history, steps_used).
    """
    if total_timesteps is None:
        total_timesteps = int(cfg.dynamics.clamp_timesteps)
    relinearize_steps = int(cfg.approx_pc.relinearize_steps or total_timesteps)
    if relinearize_steps <= 0:
        raise ValueError(
            f"approx_pc.relinearize_steps must be a positive int or null. Got: {relinearize_steps}."
        )

    remaining = total_timesteps
    combined_history = None
    while remaining > 0:
        segment_timesteps = min(relinearize_steps, remaining)
        segment_history, t, converged = run_single_linearization(model, x, rclamp, cfg, segment_timesteps)
        combined_history = concatenate_histories(combined_history, segment_history)
        remaining -= t
        if (relinearize_steps == total_timesteps) or converged:
            break

    return combined_history, total_timesteps - remaining # type: ignore

def run_pcn_clamped(model, x: jnp.ndarray, rclamp: jnp.ndarray, cfg) -> list[jnp.ndarray]:
    """Run PCN clamped phase with a single clamp to rclamp_final."""
    _, history, _ = run(model, pc_step, x,
                        max_timesteps=cfg.dynamics.clamp_timesteps,
                        tolerance=cfg.dynamics.convergence_tolerance,
                        tau=cfg.dynamics.time_constant,
                        learn=False,
                        rclamp=rclamp,
                        return_act_history=True)

    return history

def run_approx_clamped(model, x: jnp.ndarray, rclamp: jnp.ndarray, cfg) -> list[jnp.ndarray]:
    """Run approx-PC clamped phase with optional multi-nudging and re-linearization.

    Nudges toward rclamp until the model output is within
    nudge_convergence_tolerance of rclamp, or nudge_max_count is reached.
    Each nudge re-linearizes at the current model state.
    """
    total_budget = int(cfg.dynamics.clamp_timesteps)
    total_steps = 0
    combined_history = None
    current_output = model.get_activations()[-1]
    nudge_count = 0
    while (total_steps < total_budget) and (nudge_count < cfg.approx_pc.nudge.max_count):
        rnudge = current_output + (rclamp - current_output) * cfg.approx_pc.nudge.scaling_factor
        seg_history, steps_used = run_with_relinearize(model, x, rnudge, cfg, total_budget - total_steps)
        total_steps += steps_used
        current_output = model.get_activations()[-1]
        nudge_count += 1
        combined_history = concatenate_histories(combined_history, seg_history)
        if float(jnp.linalg.norm(current_output - rclamp)) < cfg.approx_pc.nudge.convergence_tolerance:
            break

    return combined_history # type: ignore

def run_single_simulation(layer_dims: list[int], cfg) -> dict:
    """Run one architecture sweep and return serialized-ready simulation data."""
    n_layers = len(layer_dims) - 1
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    # seed randomness
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    # build model
    if cfg.model.activation_fn not in ACTIVATION_FNS:
        allowed = ", ".join(sorted(ACTIVATION_FNS.keys()))
        raise ValueError(f"Unknown activation fn: {cfg.model.activation_fn}. "
                         f"Expected one of: {allowed}.")
    activation_fn = ACTIVATION_FNS[cfg.model.activation_fn]
    model = MLP(layer_dims,
                kernel_init_scale=cfg.model.kernel_init_scale,
                activation_fn=activation_fn,
                rngs=rngs)
    # for plotting later
    layer_labels = (
        [f"Hidden 1  (n={layer_dims[1]})"]
        + [f"Hidden {i + 2}  (n={layer_dims[i + 2]})" for i in range(len(layer_dims) - 3)]
        + [f"Output  (n={layer_dims[-1]})"]
    )
    # sample input
    x = jrng.normal(rngs(), (cfg.model.input_dim,))
    # run free phase with pc dynamics
    _, free_hist, t_free = run(model, pc_step, x,
                               min_timesteps=cfg.dynamics.free_min_timesteps,
                               max_timesteps=cfg.dynamics.free_max_timesteps,
                               tolerance=cfg.dynamics.convergence_tolerance,
                               tau=cfg.dynamics.time_constant,
                               learn=False,
                               return_act_history=True)
    free_hist = to_numpy(free_hist)
    print(f"Free-phase converged in {t_free} steps.")
    # get pose-free phase state
    fp_acts = model.get_activations()
    graph_def, fp_state = nnx.split(model)
    # sample noise direction
    noise_dir = jrng.normal(rngs(), fp_acts[-1].shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)

    pcn_sweep_history = []
    approx_sweep_history = []
    diff_matrix = np.zeros((n_noise, n_layers))
    convergence_tol = float(cfg.dynamics.convergence_tolerance)

    # sweep noise
    for noise_idx, magnitude in enumerate(noise_sweep):
        # build target
        rclamp_final = fp_acts[-1] + noise_dir * magnitude
        # restore model state
        pcn_model = nnx.merge(graph_def, fp_state)
        approx_model = nnx.merge(graph_def, fp_state)
        # run both models
        pcn_hist = run_pcn_clamped(pcn_model, x, rclamp_final, cfg)
        approx_hist = run_approx_clamped(approx_model, x, rclamp_final, cfg)
        pcn_sweep_history.append(to_numpy(pcn_hist))
        approx_sweep_history.append(to_numpy(approx_hist))
        # print errors
        for layer_idx in range(n_layers):
            error_val = float(jnp.linalg.norm(pcn_hist[layer_idx][-1] - approx_hist[layer_idx][-1]))
            diff_matrix[noise_idx, layer_idx] = error_val
        print(f"  noise={magnitude:.2f}  |  layer diffs: "
              + "  ".join(f"L{layer_idx + 1}={diff_matrix[noise_idx, layer_idx]:.4f}"
                          for layer_idx in range(n_layers)))

    return {
        "layer_dims": list(layer_dims),
        "layer_labels": layer_labels,
        "noise_sweep": noise_sweep,
        "relinearize_steps": cfg.approx_pc.relinearize_steps,
        "nudge": cfg.approx_pc.nudge,
        "convergence_tolerance": convergence_tol,
        "fp_output": to_numpy(fp_acts[-1]),
        "noise_dir": to_numpy(noise_dir),
        "free_history": free_hist,
        "pcn_sweep_history": pcn_sweep_history,
        "approx_sweep_history": approx_sweep_history,
        "diff_matrix": diff_matrix,
    }

def save_simulation_data(data: dict, output_file: Path) -> None:
    """Persist simulation data in a pickle file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved simulation data: {output_file}")

@hydra.main(config_path="./configs", config_name="approximation-experiment", version_base=None)
def main(cfg) -> None:
    base_dir = Path.cwd()
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    manifest = {"runs": [], "noise_sweep": noise_sweep}

    layer_sweep = cfg.experiment.get("layer_sweep", None)
    if layer_sweep is not None:
        total_layers_list = sorted(int(n) for n in layer_sweep)
        hidden_size = int(cfg.model.hidden_size)
        summary_matrix = np.zeros((n_noise, len(total_layers_list)))

        for col_idx, total_layers in enumerate(total_layers_list):
            n_hidden = total_layers - 1
            hidden_dims = [hidden_size] * n_hidden
            layer_dims = [cfg.model.input_dim, *hidden_dims, cfg.model.output_dim]
            out_dir = base_dir / f"layers_{total_layers}"

            print(f"\n{'=' * 60}")
            print(f"Running simulation: {total_layers} total layers (hidden_dims={hidden_dims})")
            print(f"{'=' * 60}")

            sim_data = run_single_simulation(layer_dims, cfg)
            data_file = out_dir / "simulation_data.pkl"
            save_simulation_data(sim_data, data_file)

            summary_matrix[:, col_idx] = sim_data["diff_matrix"].max(axis=1)
            manifest["runs"].append({"total_layers": int(total_layers),
                                     "out_dir": str(out_dir.relative_to(base_dir)),
                                     "data_file": str(data_file.relative_to(base_dir))})

        summary_file = base_dir / "sweep_summary.npz"
        np.savez_compressed(summary_file,
                            summary_matrix=summary_matrix,
                            total_layers_list=np.asarray(total_layers_list),
                            noise_sweep=np.asarray(noise_sweep),
                            convergence_tolerance=np.asarray(float(cfg.dynamics.convergence_tolerance)))
        manifest["summary_file"] = str(summary_file.relative_to(base_dir)) # type: ignore
        print(f"\nSaved summary data: {summary_file}")
    else:
        layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
        out_dir = base_dir / "single_run"
        sim_data = run_single_simulation(layer_dims, cfg)
        data_file = out_dir / "simulation_data.pkl"
        save_simulation_data(sim_data, data_file)

        manifest["runs"].append({"total_layers": int(len(layer_dims) - 1),
                                 "out_dir": str(out_dir.relative_to(base_dir)),
                                 "data_file": str(data_file.relative_to(base_dir))})

    manifest_file = base_dir / "approximation_experiment_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_file}")

if __name__ == "__main__":
    main()
