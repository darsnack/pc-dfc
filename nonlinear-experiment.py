"""Run the nonlinear relinearization experiment for Figures 4 and 5.

Run nonlinear-plotting.py separately to plot the saved simulation_data.pkl.
"""
import pickle
from pathlib import Path

import hydra
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
from flax import nnx
from omegaconf import OmegaConf
from tqdm import tqdm

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run
from projectlib.models import MLP


# run() supplies (model, x); the step functions also take an optimizer.
def pc_dynamics(model, x, **kwargs):
    return pc_step(model, None, x, **kwargs)


def approx_dynamics(model, x, **kwargs):
    return approx_pc_step(model, None, x, **kwargs)


def run_approx(model, x, rclamp, cfg, relinearize_steps, pbar):
    """Run Approx PC, stopping when a segment converges (None freezes the VJP)."""
    total_timesteps = int(cfg.dynamics.clamp_timesteps)
    step_size = relinearize_steps or total_timesteps
    remaining = total_timesteps
    history = None
    n_vjp = 0
    segment_stats = []
    while remaining > 0:
        n = min(step_size, remaining)
        vjps = current_pc_vjp(model, x)
        n_vjp += 1
        # The paper sweep uses five steps minimum, including the rs=1 condition.
        _, _, seg, t = run(model, approx_dynamics, x,
                           vjps=vjps, k=cfg.approx_pc.k,
                           min_timesteps=5, max_timesteps=n,
                           tolerance=cfg.dynamics.convergence_tolerance,
                           tau=cfg.dynamics.time_constant,
                           learn=False, rclamp=rclamp,
                           return_act_history=True,
                           disable_tqdm=True)
        dvs = jnp.array([jnp.sum((act[-1] - act[-2]) ** 2, axis=-1) for act in seg])
        speed = float(jnp.min(jnp.sqrt(jnp.sum(dvs, axis=0))))
        converged = speed < float(cfg.dynamics.convergence_tolerance)
        segment_stats.append({
            "speed": speed,
            "converged": converged,
            "steps": int(t),
            "max_steps": int(n),
        })
        history = seg if history is None else [
            jnp.concatenate([old, new[1:]], axis=0) for old, new in zip(history, seg)
        ]
        remaining -= t
        pbar.update(t)
        if (step_size == total_timesteps) or converged:
            break
    n_relinearizations = max(0, n_vjp - 1)
    return [np.asarray(h) for h in history], n_relinearizations, segment_stats


@hydra.main(config_path="./configs", config_name="nonlinear-experiment", version_base=None)
def main(cfg) -> None:
    # Build model
    layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
    n_layers = len(layer_dims) - 1
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    model = MLP(layer_dims, kernel_init_scale=cfg.model.kernel_init_scale,
                activation_fn=nnx.tanh, use_bias=False, rngs=rngs)

    # Free phase
    rng = jrng.PRNGKey(cfg.seed)
    rng, rng_x, rng_n = jrng.split(rng, 3)
    x = jrng.normal(rng_x, (cfg.model.input_dim,))
    _, _, free_history, t_free = run(model, pc_dynamics, x,
                                     min_timesteps=cfg.dynamics.free_min_timesteps,
                                     max_timesteps=cfg.dynamics.free_max_timesteps,
                                     tolerance=cfg.dynamics.convergence_tolerance,
                                     tau=cfg.dynamics.time_constant,
                                     learn=False, return_act_history=True)
    print(f"Free phase converged in {t_free} steps.")

    fp_acts = model.get_activations()
    graph_def, fp_state = nnx.split(model)

    initial_vjps = current_pc_vjp(model, x)

    # Clamp target
    noise_dir = jrng.normal(rng_n, fp_acts[-1].shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)
    rclamp = fp_acts[-1] + noise_dir * cfg.noise

    # Run PCN
    pcn_model = nnx.merge(graph_def, fp_state)
    _, _, pcn_history, _ = run(pcn_model, pc_dynamics, x,
                               max_timesteps=cfg.dynamics.clamp_timesteps,
                               tolerance=cfg.dynamics.convergence_tolerance,
                               tau=cfg.dynamics.time_constant,
                               learn=False, rclamp=rclamp,
                               return_act_history=True)
    pcn_hist = [np.asarray(h) for h in pcn_history]
    print("PCN done.")

    # Run approx-PC for each relinearize_steps
    relinearize_steps_sweep = [None if rs is None else int(rs)
                                for rs in cfg.relinearize_steps_sweep]
    approx_hists = {}
    approx_relin_counts = {}
    approx_segment_stats = {}
    for rs in relinearize_steps_sweep:
        with tqdm(total=int(cfg.dynamics.clamp_timesteps),
                  desc=f"approx-PC  relinearize_steps={rs}") as pbar:
            approx_model = nnx.merge(graph_def, fp_state)
            hist, n_relin, segment_stats = run_approx(approx_model, x, rclamp, cfg, rs, pbar=pbar)
            approx_hists[rs] = hist
            approx_relin_counts[rs] = n_relin
            approx_segment_stats[rs] = segment_stats

    # Measure the change in flow along the frozen-linearization trajectory.
    T = approx_hists[None][0].shape[0]
    delta_flow = np.zeros((T, n_layers))

    for t in range(T):
        model_local = nnx.merge(graph_def, fp_state)
        for l in range(n_layers):
            model_local.activations[l].value = jnp.asarray(approx_hists[None][l][t])
        model_local.activations[-1].value = jnp.asarray(rclamp)
        local_vjps = current_pc_vjp(model_local, x)

        v_before_local = [np.asarray(a.value) for a in model_local.activations]
        approx_pc_step(model_local, None, x, vjps=local_vjps, k=cfg.approx_pc.k,
                       tau=cfg.dynamics.time_constant, rclamp=rclamp, learn=False)
        dv_local = [np.asarray(a.value) - before
                    for a, before in zip(model_local.activations, v_before_local)]

        model_init = nnx.merge(graph_def, fp_state)
        for l in range(n_layers):
            model_init.activations[l].value = jnp.asarray(approx_hists[None][l][t])
        model_init.activations[-1].value = jnp.asarray(rclamp)

        v_before_init = [np.asarray(a.value) for a in model_init.activations]
        approx_pc_step(model_init, None, x, vjps=initial_vjps, k=cfg.approx_pc.k,
                       tau=cfg.dynamics.time_constant, rclamp=rclamp, learn=False)
        dv_init = [np.asarray(a.value) - before
                   for a, before in zip(model_init.activations, v_before_init)]

        for layer_idx in range(n_layers):
            delta_flow[t, layer_idx] = np.linalg.norm(dv_local[layer_idx] - dv_init[layer_idx])

    data = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "pcn_hist": pcn_hist,
        "free_history": [np.asarray(h) for h in free_history],
        "fp_output": np.asarray(fp_acts[-1]),
        "noise_dir": np.asarray(noise_dir),
        "approx_hists": approx_hists,
        "approx_relin_counts": approx_relin_counts,
        "approx_segment_stats": approx_segment_stats,
        "delta_flow": delta_flow,
    }
    output = Path.cwd() / "simulation_data.pkl"
    with output.open("wb") as f:
        pickle.dump(data, f)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
