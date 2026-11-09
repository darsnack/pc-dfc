"""Generate the linear depth and perturbation sweeps for Figures 2 and 3."""

import json
import pickle
from pathlib import Path

import hydra
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from flax import nnx

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run
from projectlib.models import MLP
from projectlib.utils import noop

# drop the optimizer for these inference only experiments
def pc_dynamics(model, x, **kwargs):
    return pc_step(model, None, x, **kwargs)

def approx_dynamics(model, x, **kwargs):
    return approx_pc_step(model, None, x, **kwargs)


def run_single_model(layer_dims: list[int], cfg) -> dict:
    """Run one architecture sweep and return serialized-ready simulation data."""
    n_layers = len(layer_dims) - 1
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    # seed randomness
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    model = MLP(layer_dims,
                kernel_init_scale=cfg.model.kernel_init_scale,
                activation_fn=noop,
                use_bias=False,
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
    _, _, free_hist, t_free = run(
        model, pc_dynamics, x,
        return_act_history=True,
        min_timesteps=cfg.dynamics.free_min_timesteps,
        max_timesteps=cfg.dynamics.free_max_timesteps,
        tolerance=cfg.dynamics.convergence_tolerance,
        tau=cfg.dynamics.time_constant,
        learn=False,
    )
    print(f"Free-phase converged in {t_free} steps.")
    # Save the free-phase state for both clamped dynamics.
    fp_acts = model.get_activations()
    graph_def, fp_state = nnx.split(model)
    # sample noise direction
    noise_dir = jrng.normal(rngs(), fp_acts[-1].shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)

    pcn_sweep_history = []
    approx_sweep_history = []
    diff_matrix = np.zeros((n_noise, n_layers))

    # sweep noise
    for noise_idx, magnitude in enumerate(noise_sweep):
        # build target
        rclamp_final = fp_acts[-1] + noise_dir * magnitude
        # restore model state
        pcn_model = nnx.merge(graph_def, fp_state)
        approx_model = nnx.merge(graph_def, fp_state)
        # run both models
        _, _, pcn_hist, _ = run(
            pcn_model, pc_dynamics, x,
            max_timesteps=cfg.dynamics.clamp_timesteps,
            tolerance=cfg.dynamics.convergence_tolerance,
            tau=cfg.dynamics.time_constant,
            learn=False,
            rclamp=rclamp_final,
            return_act_history=True,
        )
        vjps = current_pc_vjp(approx_model, x)
        _, _, approx_hist, _ = run(
            approx_model, approx_dynamics, x,
            vjps=vjps,
            k=cfg.approx_pc.k,
            max_timesteps=cfg.dynamics.clamp_timesteps,
            tolerance=cfg.dynamics.convergence_tolerance,
            tau=cfg.dynamics.time_constant,
            learn=False,
            rclamp=rclamp_final,
            return_act_history=True,
        )
        pcn_sweep_history.append([np.asarray(hist) for hist in pcn_hist])
        approx_sweep_history.append([np.asarray(hist) for hist in approx_hist])
        # print errors
        for layer_idx in range(n_layers):
            error_val = float(jnp.linalg.norm(pcn_hist[layer_idx][-1] - approx_hist[layer_idx][-1]))
            diff_matrix[noise_idx, layer_idx] = error_val
        print(f"  noise={magnitude:.2f}  |  layer diffs: "
              + "  ".join(f"L{layer_idx + 1}={diff_matrix[noise_idx, layer_idx]:.4f}"
                          for layer_idx in range(n_layers)))

    return {
        "free_history": [np.asarray(hist) for hist in free_hist],
        "fp_output": np.asarray(fp_acts[-1]),
        "noise_dir": np.asarray(noise_dir),
        "layer_dims": list(layer_dims),
        "layer_labels": layer_labels,
        "noise_sweep": noise_sweep,
        "convergence_tolerance": float(cfg.dynamics.convergence_tolerance),
        "pcn_sweep_history": pcn_sweep_history,
        "approx_sweep_history": approx_sweep_history,
        "diff_matrix": diff_matrix,
    }

@hydra.main(config_path="./configs", config_name="linear-experiment", version_base=None)
def main(cfg) -> None:
    base_dir = Path.cwd()
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    manifest = {"runs": [], "noise_sweep": noise_sweep}

    total_layers_list = sorted(int(n) for n in cfg.experiment.layer_sweep)
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

        sim_data = run_single_model(layer_dims, cfg)
        data_file = out_dir / "simulation_data.pkl"
        out_dir.mkdir(parents=True, exist_ok=True)
        with data_file.open("wb") as f:
            pickle.dump(sim_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved simulation data: {data_file}")

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
    manifest["summary_file"] = str(summary_file.relative_to(base_dir))
    print(f"\nSaved summary data: {summary_file}")
    manifest_file = base_dir / "linear_experiment_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_file}")

if __name__ == "__main__":
    main()
