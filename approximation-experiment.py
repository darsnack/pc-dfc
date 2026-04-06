"""Run linear approximation experiments and save raw outputs to disk.

This script runs PCN vs approx-PC dynamics sweeps and persists simulation
artifacts. Plot generation is intentionally separated into
`linear-approx-plotting.py`.
"""

from __future__ import annotations

import json
import pickle
from functools import partial
from pathlib import Path

import hydra
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from flax import nnx

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run
from projectlib.models import MLP
from projectlib.utils import noop


def concat(free: jnp.ndarray, clamped: jnp.ndarray) -> jnp.ndarray:
    """Stitch free-phase and clamped-phase histories (drop duplicate junction)."""
    return jnp.concatenate([free, clamped[1:]], axis=0)


def run_single_linearized_approximation(
    model: nnx.Module,
    x: jnp.ndarray,
    rclamp: jnp.ndarray,
    cfg,
    timesteps: int,
) -> list[jnp.ndarray]:
    """Run approx-PC for a fixed number of steps with one linearization."""
    vjps = current_pc_vjp(model, x)
    approx_step = partial(approx_pc_step, vjps=vjps, k=cfg.approx_pc.k)
    _, approx_ch, _ = run(
        model,
        approx_step,
        x,
        min_timesteps=timesteps,
        max_timesteps=timesteps,
        tolerance=cfg.dynamics.convergence_tolerance,
        tau=cfg.dynamics.time_constant,
        learn=False,
        rclamp=rclamp,
        return_act_history=True,
    )

    return approx_ch


def run_relinearized_approximation(
    model: nnx.Module,
    x: jnp.ndarray,
    rclamp: jnp.ndarray,
    cfg,
) -> list[jnp.ndarray]:
    """Run approx-PC while optionally re-linearizing every fixed number of steps."""
    total_timesteps = int(cfg.dynamics.clamp_timesteps)
    steps_between_linearizations = cfg.approx_pc.get("steps_between_linearizations", None)

    if steps_between_linearizations is None:
        return run_single_linearized_approximation(model, x, rclamp, cfg, total_timesteps)

    steps_between_linearizations = int(steps_between_linearizations)
    if steps_between_linearizations <= 0:
        raise ValueError(
            "approx_pc.steps_between_linearizations must be a positive int or null. "
            f"Got: {steps_between_linearizations}."
        )

    remaining = total_timesteps
    combined_history = None
    while remaining > 0:
        segment_timesteps = min(steps_between_linearizations, remaining)
        segment_history = run_single_linearized_approximation(
            model,
            x,
            rclamp,
            cfg,
            segment_timesteps,
        )
        if combined_history is None:
            combined_history = segment_history
        else:
            combined_history = [
                concat(prev_hist, next_hist)
                for prev_hist, next_hist in zip(combined_history, segment_history)
            ]
        remaining -= segment_timesteps

    return combined_history


def run_single_simulation(layer_dims: list[int], cfg) -> dict:
    """Run one architecture sweep and return serialized-ready simulation data."""
    n_layers = len(layer_dims) - 1
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)
    steps_between_linearizations = cfg.approx_pc.get("steps_between_linearizations", None)

    activation_fns = {
        "identity": noop,
        "noop": noop,
        "tanh": nnx.tanh,
        "relu": nnx.relu,
    }
    if cfg.model.activation_fn not in activation_fns:
        allowed = ", ".join(sorted(activation_fns.keys()))
        raise ValueError(
            f"Unknown activation fn: {cfg.model.activation_fn}. "
            f"Expected one of: {allowed}."
        )
    activation_fn = activation_fns[cfg.model.activation_fn]

    layer_labels = (
        [f"Hidden 1  (n={layer_dims[1]})"]
        + [f"Hidden {i + 2}  (n={layer_dims[i + 2]})" for i in range(len(layer_dims) - 3)]
        + [f"Output  (n={layer_dims[-1]})"]
    )

    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    model = MLP(
        layer_dims,
        kernel_init_scale=cfg.model.kernel_init_scale,
        activation_fn=activation_fn,
        rngs=rngs,
    )

    rng = jrng.PRNGKey(cfg.seed)
    rng, rng_x, rng_n = jrng.split(rng, 3)
    x = jrng.normal(rng_x, (cfg.model.input_dim,))

    _, free_hist, t_free = run(
        model,
        pc_step,
        x,
        min_timesteps=cfg.dynamics.free_min_timesteps,
        max_timesteps=cfg.dynamics.free_max_timesteps,
        tolerance=cfg.dynamics.convergence_tolerance,
        tau=cfg.dynamics.time_constant,
        learn=False,
        return_act_history=True,
    )
    print(f"Free-phase converged in {t_free} steps.")

    fp_acts = model.get_activations()
    graph_def, fp_state = nnx.split(model)

    noise_dir = jrng.normal(rng_n, fp_acts[-1].shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)

    sweep_pcn_clamp = []
    sweep_approx_clamp = []
    sweep_pcn_full = []
    sweep_approx_full = []
    diff_matrix = np.zeros((n_noise, n_layers))
    convergence_tol = float(cfg.dynamics.convergence_tolerance)

    free_hist_np = [np.asarray(h) for h in free_hist]

    for noise_idx, magnitude in enumerate(noise_sweep):
        rclamp = fp_acts[-1] + noise_dir * magnitude

        pcn_model = nnx.merge(graph_def, fp_state)
        approx_model = nnx.merge(graph_def, fp_state)

        _, pcn_ch, _ = run(
            pcn_model,
            pc_step,
            x,
            min_timesteps=cfg.dynamics.clamp_timesteps,
            max_timesteps=cfg.dynamics.clamp_timesteps,
            tolerance=cfg.dynamics.convergence_tolerance,
            tau=cfg.dynamics.time_constant,
            learn=False,
            rclamp=rclamp,
            return_act_history=True,
        )
        approx_ch = run_relinearized_approximation(approx_model, x, rclamp, cfg)

        pcn_ch_np = [np.asarray(h) for h in pcn_ch]
        approx_ch_np = [np.asarray(h) for h in approx_ch]

        sweep_pcn_clamp.append(pcn_ch_np)
        sweep_approx_clamp.append(approx_ch_np)
        sweep_pcn_full.append(
            [np.asarray(concat(free, clamped)) for free, clamped in zip(free_hist_np, pcn_ch_np)]
        )
        sweep_approx_full.append(
            [
                np.asarray(concat(free, clamped))
                for free, clamped in zip(free_hist_np, approx_ch_np)
            ]
        )

        for layer_idx in range(n_layers):
            error_val = np.linalg.norm(
                sweep_pcn_clamp[-1][layer_idx][-1] - sweep_approx_clamp[-1][layer_idx][-1]
            )
            diff_matrix[noise_idx, layer_idx] = max(error_val, convergence_tol)

        print(
            f"  noise={magnitude:.2f}  |  layer diffs: "
            + "  ".join(
                f"L{layer_idx + 1}={diff_matrix[noise_idx, layer_idx]:.4f}"
                for layer_idx in range(n_layers)
            )
        )

    return {
        "layer_dims": list(layer_dims),
        "layer_labels": layer_labels,
        "noise_sweep": noise_sweep,
        "free_timesteps": int(t_free),
        "clamp_timesteps": int(cfg.dynamics.clamp_timesteps),
        "steps_between_linearizations": (
            None
            if steps_between_linearizations is None
            else int(steps_between_linearizations)
        ),
        "convergence_tolerance": convergence_tol,
        "fp_output": np.asarray(fp_acts[-1]),
        "noise_dir": np.asarray(noise_dir),
        "sweep_pcn_clamp": sweep_pcn_clamp,
        "sweep_approx_clamp": sweep_approx_clamp,
        "sweep_pcn_full": sweep_pcn_full,
        "sweep_approx_full": sweep_approx_full,
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
    manifest = {
        "runs": [],
        "noise_sweep": noise_sweep,
    }

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
            print(
                f"Running simulation: {total_layers} total layers  "
                f"(hidden_dims={hidden_dims})"
            )
            print(f"{'=' * 60}")

            sim_data = run_single_simulation(layer_dims, cfg)
            data_file = out_dir / "simulation_data.pkl"
            save_simulation_data(sim_data, data_file)

            summary_matrix[:, col_idx] = sim_data["diff_matrix"].max(axis=1)
            manifest["runs"].append(
                {
                    "total_layers": int(total_layers),
                    "out_dir": str(out_dir.relative_to(base_dir)),
                    "data_file": str(data_file.relative_to(base_dir)),
                }
            )

        summary_file = base_dir / "sweep_summary.npz"
        np.savez_compressed(
            summary_file,
            summary_matrix=summary_matrix,
            total_layers_list=np.asarray(total_layers_list),
            noise_sweep=np.asarray(noise_sweep),
            convergence_tolerance=np.asarray(float(cfg.dynamics.convergence_tolerance)),
        )
        manifest["summary_file"] = str(summary_file.relative_to(base_dir))
        print(f"\nSaved summary data: {summary_file}")
    else:
        layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
        out_dir = base_dir / "single_run"
        sim_data = run_single_simulation(layer_dims, cfg)
        data_file = out_dir / "simulation_data.pkl"
        save_simulation_data(sim_data, data_file)

        manifest["runs"].append(
            {
                "total_layers": int(len(layer_dims) - 1),
                "out_dir": str(out_dir.relative_to(base_dir)),
                "data_file": str(data_file.relative_to(base_dir)),
            }
        )

    manifest_file = base_dir / "approximation_experiment_manifest.json"
    with manifest_file.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: {manifest_file}")


if __name__ == "__main__":
    main()
