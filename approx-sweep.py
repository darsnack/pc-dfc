"""Sweep approximation error over noise, layer count, relinearize_steps, and scaling_factor."""
import json
import pickle
import hydra
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from pathlib import Path
from flax import nnx
from tqdm import tqdm

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run, is_converged
from projectlib.models import MLP
from projectlib.utils import noop

ACTIVATION_FNS = {
    "identity": noop,
    "noop": noop,
    "tanh": nnx.tanh,
    "relu": nnx.relu,
}


def concatenate_histories(old, new):
    if old is None:
        return new
    return [jnp.concatenate([o, n[1:]], axis=0) for o, n in zip(old, new)]


def to_numpy(x):
    if isinstance(x, (tuple, list)):
        return type(x)(to_numpy(xi) for xi in x)
    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}
    elif isinstance(x, jnp.ndarray):
        return np.asarray(x)
    else:
        return x


def run_single_linearization(model, x, rclamp, cfg, timesteps):
    vjps = current_pc_vjp(model, x)
    _, history, t = run(model, approx_pc_step, x,
                        vjps=vjps,
                        k=cfg.approx_pc.k,
                        max_timesteps=timesteps,
                        tolerance=cfg.dynamics.convergence_tolerance,
                        tau=cfg.dynamics.time_constant,
                        learn=False,
                        rclamp=rclamp,
                        return_act_history=True,
                        disable_tqdm=True)
    converged = is_converged(history, cfg.dynamics.convergence_tolerance, "norm")
    return history, t, converged


def run_with_relinearize(model, x, rclamp, cfg, total_timesteps, relinearize_steps):
    step_size = int(relinearize_steps or total_timesteps)
    remaining = total_timesteps
    combined_history = None
    while remaining > 0:
        n = min(step_size, remaining)
        seg_history, t, converged = run_single_linearization(model, x, rclamp, cfg, n)
        combined_history = concatenate_histories(combined_history, seg_history)
        remaining -= t
        if (step_size == total_timesteps) or converged:
            break
    return combined_history, total_timesteps - remaining


def run_pcn_clamped(model, x, rclamp, cfg):
    _, history, _ = run(model, pc_step, x,
                        max_timesteps=cfg.dynamics.clamp_timesteps,
                        tolerance=cfg.dynamics.convergence_tolerance,
                        tau=cfg.dynamics.time_constant,
                        learn=False,
                        rclamp=rclamp,
                        return_act_history=True,
                        disable_tqdm=True)
    return history


def run_approx_clamped(model, x, rclamp, cfg, relinearize_steps, scaling_factor, pbar=None):
    total_budget = int(cfg.dynamics.clamp_timesteps)
    total_steps = 0
    combined_history = None
    current_output = model.get_activations()[-1]
    nudge_count = 0
    nudge_step_counts = []
    while (total_steps < total_budget) and (nudge_count < cfg.approx_pc.nudge.max_count):
        rnudge = current_output + (rclamp - current_output) * scaling_factor
        seg_history, steps_used = run_with_relinearize(
            model, x, rnudge, cfg, total_budget - total_steps, relinearize_steps
        )
        total_steps += steps_used
        current_output = model.get_activations()[-1]
        nudge_count += 1
        nudge_step_counts.append(steps_used)
        combined_history = concatenate_histories(combined_history, seg_history)
        if pbar is not None:
            pbar.update(1)
        if float(jnp.linalg.norm(current_output - rclamp)) < cfg.approx_pc.nudge.convergence_tolerance:
            break
    return combined_history, nudge_count, nudge_step_counts


def run_single_simulation(layer_dims, cfg):
    n_layers = len(layer_dims) - 1
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    relinearize_steps_sweep = [None if rs is None else int(rs)
                                for rs in cfg.experiment.relinearize_steps_sweep]
    scaling_factor_sweep = [float(sf) for sf in cfg.experiment.scaling_factor_sweep]
    n_rs = len(relinearize_steps_sweep)
    n_sf = len(scaling_factor_sweep)

    # seed and build model
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    if cfg.model.activation_fn not in ACTIVATION_FNS:
        raise ValueError(f"Unknown activation_fn: {cfg.model.activation_fn}")
    activation_fn = ACTIVATION_FNS[cfg.model.activation_fn]
    model = MLP(layer_dims, kernel_init_scale=cfg.model.kernel_init_scale,
                activation_fn=activation_fn, rngs=rngs)
    layer_labels = (
        [f"Hidden 1  (n={layer_dims[1]})"]
        + [f"Hidden {i+2}  (n={layer_dims[i+2]})" for i in range(len(layer_dims) - 3)]
        + [f"Output  (n={layer_dims[-1]})"]
    )

    # free phase
    x = jrng.normal(rngs(), (cfg.model.input_dim,))
    _, free_hist, t_free = run(model, pc_step, x,
                               min_timesteps=cfg.dynamics.free_min_timesteps,
                               max_timesteps=cfg.dynamics.free_max_timesteps,
                               tolerance=cfg.dynamics.convergence_tolerance,
                               tau=cfg.dynamics.time_constant,
                               learn=False,
                               return_act_history=True)
    free_hist = to_numpy(free_hist)
    print(f"Free-phase converged in {t_free} steps.")
    fp_acts = model.get_activations()
    graph_def, fp_state = nnx.split(model)
    noise_dir = jrng.normal(rngs(), fp_acts[-1].shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)

    # results: pcn indexed by [noise_idx]
    #          approx indexed by [rs_idx][sf_idx][noise_idx]
    pcn_sweep_history = []
    approx_sweep_history = [
        [[] for _ in range(n_sf)]
        for _ in range(n_rs)
    ]
    approx_nudge_steps = [
        [[] for _ in range(n_sf)]
        for _ in range(n_rs)
    ]
    approx_nudge_counts = [
        [[] for _ in range(n_sf)]
        for _ in range(n_rs)
    ]
    diff_matrix = np.zeros((n_rs, n_sf, n_noise, n_layers))

    for noise_idx, magnitude in enumerate(noise_sweep):
        rclamp_final = fp_acts[-1] + noise_dir * magnitude

        pcn_model = nnx.merge(graph_def, fp_state)
        pcn_hist = run_pcn_clamped(pcn_model, x, rclamp_final, cfg)
        pcn_sweep_history.append(to_numpy(pcn_hist))

        for rs_idx, rs in enumerate(relinearize_steps_sweep):
            for sf_idx, sf in enumerate(scaling_factor_sweep):
                rs_str = "null" if rs is None else str(rs)
                desc = f"noise={magnitude:.2f}  rs={rs_str}  sf={sf:.2f}"
                with tqdm(total=cfg.approx_pc.nudge.max_count, desc=desc, unit="nudge") as pbar:
                    approx_model = nnx.merge(graph_def, fp_state)
                    approx_hist, nudge_count, nudge_steps = run_approx_clamped(approx_model, x, rclamp_final, cfg, rs, sf, pbar=pbar)
                approx_sweep_history[rs_idx][sf_idx].append(to_numpy(approx_hist))
                approx_nudge_steps[rs_idx][sf_idx].append(nudge_steps)
                approx_nudge_counts[rs_idx][sf_idx].append(nudge_count)
                for layer_idx in range(n_layers):
                    err = float(jnp.linalg.norm(
                        pcn_hist[layer_idx][-1] - approx_hist[layer_idx][-1]
                    ))
                    diff_matrix[rs_idx, sf_idx, noise_idx, layer_idx] = err
                rs_str = "null" if rs is None else str(rs)
                print(f"  noise={magnitude:.2f}  rs={rs_str}  sf={sf:.2f}  |  "
                      + "  ".join(f"L{i+1}={diff_matrix[rs_idx, sf_idx, noise_idx, i]:.4f}"
                                  for i in range(n_layers)))

    return {
        "layer_dims": list(layer_dims),
        "layer_labels": layer_labels,
        "noise_sweep": noise_sweep,
        "relinearize_steps_sweep": relinearize_steps_sweep,
        "scaling_factor_sweep": scaling_factor_sweep,
        "convergence_tolerance": float(cfg.dynamics.convergence_tolerance),
        "fp_output": to_numpy(fp_acts[-1]),
        "noise_dir": to_numpy(noise_dir),
        "free_history": free_hist,
        "pcn_sweep_history": pcn_sweep_history,
        "approx_sweep_history": approx_sweep_history,
        "approx_nudge_steps": approx_nudge_steps,
        "approx_nudge_counts": approx_nudge_counts,
        "diff_matrix": diff_matrix,
    }


def save_simulation_data(data, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved simulation data: {output_file}")


@hydra.main(config_path="./configs", config_name="approx-sweep", version_base=None)
def main(cfg):
    base_dir = Path.cwd()
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    relinearize_steps_sweep = [None if rs is None else int(rs)
                                for rs in cfg.experiment.relinearize_steps_sweep]
    scaling_factor_sweep = [float(sf) for sf in cfg.experiment.scaling_factor_sweep]
    n_rs = len(relinearize_steps_sweep)
    n_sf = len(scaling_factor_sweep)

    manifest = {
        "runs": [],
        "noise_sweep": noise_sweep,
        "relinearize_steps_sweep": relinearize_steps_sweep,
        "scaling_factor_sweep": scaling_factor_sweep,
    }

    layer_sweep = cfg.experiment.get("layer_sweep", None)
    if layer_sweep is not None:
        total_layers_list = sorted(int(n) for n in layer_sweep)
        hidden_size = int(cfg.model.hidden_size)
        # summary_matrix[rs_idx, sf_idx, noise_idx, layer_col_idx] = max diff across MLP layers
        summary_matrix = np.zeros((n_rs, n_sf, n_noise, len(total_layers_list)))

        for col_idx, total_layers in enumerate(total_layers_list):
            n_hidden = total_layers - 1
            hidden_dims = [hidden_size] * n_hidden
            layer_dims = [cfg.model.input_dim, *hidden_dims, cfg.model.output_dim]
            out_dir = base_dir / f"layers_{total_layers}"

            print(f"\n{'=' * 60}")
            print(f"Running: {total_layers} total layers (hidden_dims={hidden_dims})")
            print(f"{'=' * 60}")

            sim_data = run_single_simulation(layer_dims, cfg)
            data_file = out_dir / "simulation_data.pkl"
            save_simulation_data(sim_data, data_file)

            # diff_matrix[rs_idx, sf_idx, noise_idx, layer_idx] → max over layer_idx
            summary_matrix[:, :, :, col_idx] = sim_data["diff_matrix"].max(axis=-1)
            manifest["runs"].append({
                "total_layers": int(total_layers),
                "out_dir": str(out_dir.relative_to(base_dir)),
                "data_file": str(data_file.relative_to(base_dir)),
            })

        summary_file = base_dir / "sweep_summary.npz"
        np.savez_compressed(
            summary_file,
            summary_matrix=summary_matrix,
            total_layers_list=np.asarray(total_layers_list),
            noise_sweep=np.asarray(noise_sweep),
            # encode null as -1 for npz storage
            relinearize_steps_sweep=np.asarray([-1 if rs is None else rs
                                                 for rs in relinearize_steps_sweep]),
            scaling_factor_sweep=np.asarray(scaling_factor_sweep),
        )
        manifest["summary_file"] = str(summary_file.relative_to(base_dir))
        print(f"\nSaved summary: {summary_file}")
    else:
        layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
        out_dir = base_dir / "single_run"
        sim_data = run_single_simulation(layer_dims, cfg)
        data_file = out_dir / "simulation_data.pkl"
        save_simulation_data(sim_data, data_file)
        manifest["runs"].append({
            "total_layers": int(len(layer_dims) - 1),
            "out_dir": str(out_dir.relative_to(base_dir)),
            "data_file": str(data_file.relative_to(base_dir)),
        })

    manifest_file = base_dir / "approximation_sweeps_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_file}")


if __name__ == "__main__":
    main()
