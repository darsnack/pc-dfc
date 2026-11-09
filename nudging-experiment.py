"""Run the fixed six-layer nudging experiment for Figure 6.

Save simulation_data.pkl for nudging-plotting.py; no figures are generated here.
"""
import pickle
from pathlib import Path

import hydra
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from flax import nnx
from tqdm import tqdm

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run
from projectlib.models import MLP


def pc_dynamics(model, x, **kwargs):
    return pc_step(model, None, x, **kwargs)


def approx_dynamics(model, x, **kwargs):
    return approx_pc_step(model, None, x, **kwargs)


@hydra.main(config_path="./configs", config_name="nudging-experiment", version_base=None)
def main(cfg):
    layer_dims = [1, 8, 8, 8, 8, 8, 1]
    layer_labels = [f"Hidden {i + 1}  (n=8)" for i in range(5)] + ["Output  (n=1)"]
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    model = MLP(layer_dims, kernel_init_scale=1.0, activation_fn=nnx.tanh,
                use_bias=False, rngs=rngs)
    x = jrng.normal(rngs(), (1,))
    _, _, free_history, t_free = run(
        model, pc_dynamics, x,
        min_timesteps=cfg.dynamics.free_min_timesteps,
        max_timesteps=cfg.dynamics.free_max_timesteps,
        tolerance=cfg.dynamics.convergence_tolerance,
        tau=cfg.dynamics.time_constant, learn=False, return_act_history=True,
    )
    print(f"Free-phase converged in {t_free} steps.")
    fp_output = model.get_activations()[-1]
    graph_def, fp_state = nnx.split(model)
    noise_dir = jrng.normal(rngs(), fp_output.shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)
    rclamp = fp_output + noise_dir  # Unit perturbation, as in Figure 6.

    pcn_model = nnx.merge(graph_def, fp_state)
    _, _, pcn_history, _ = run(
        pcn_model, pc_dynamics, x, rclamp=rclamp,
        max_timesteps=cfg.dynamics.clamp_timesteps,
        tolerance=cfg.dynamics.convergence_tolerance,
        tau=cfg.dynamics.time_constant, learn=False,
        return_act_history=True, disable_tqdm=True,
    )

    scaling_factors = list(cfg.scaling_factors)
    approx_histories, nudge_steps, final_errors = [], [], []
    for scaling_factor in scaling_factors:
        approx_model = nnx.merge(graph_def, fp_state)
        history = None
        step_counts = []
        total_steps = 0
        current_output = approx_model.get_activations()[-1]
        with tqdm(total=cfg.nudge.max_count, desc=f"sf={scaling_factor:.2f}", unit="nudge") as pbar:
            while total_steps < cfg.dynamics.clamp_timesteps and len(step_counts) < cfg.nudge.max_count:
                rnudge = current_output + (rclamp - current_output) * scaling_factor
                # Keep this linearization until convergence before nudging again.
                vjps = current_pc_vjp(approx_model, x)
                _, _, segment, steps_used = run(
                    approx_model, approx_dynamics, x, vjps=vjps,
                    k=1.0, rclamp=rnudge,
                    max_timesteps=cfg.dynamics.clamp_timesteps - total_steps,
                    tolerance=cfg.dynamics.convergence_tolerance,
                    tau=cfg.dynamics.time_constant, learn=False,
                    return_act_history=True, disable_tqdm=True,
                )
                total_steps += steps_used
                step_counts.append(steps_used)
                history = segment if history is None else [
                    jnp.concatenate([old, new[1:]], axis=0)
                    for old, new in zip(history, segment)
                ]
                current_output = approx_model.get_activations()[-1]
                pbar.update(1)
                if float(jnp.linalg.norm(current_output - rclamp)) < cfg.nudge.convergence_tolerance:
                    break
        approx_histories.append([np.asarray(layer) for layer in history])
        nudge_steps.append(step_counts)
        errors = [float(jnp.linalg.norm(pc[-1] - ap[-1])) for pc, ap in zip(pcn_history, history)]
        final_errors.append(errors)
        print(f"sf={scaling_factor:.2f}  |  " + "  ".join(f"L{i + 1}={err:.4f}" for i, err in enumerate(errors)))

    data = {
        "layer_dims": layer_dims,
        "layer_labels": layer_labels,
        "scaling_factors": scaling_factors,
        "nudge_convergence_tolerance": float(cfg.nudge.convergence_tolerance),
        "free_history": [np.asarray(layer) for layer in free_history],
        "fp_output": np.asarray(fp_output),
        "noise_dir": np.asarray(noise_dir),
        "pcn_history": [np.asarray(layer) for layer in pcn_history],
        "approx_histories": approx_histories,
        "nudge_steps": nudge_steps,
        "final_errors": np.asarray(final_errors),
    }
    output_file = Path.cwd() / "simulation_data.pkl"
    with output_file.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved simulation data: {output_file}")


if __name__ == "__main__":
    main()
