"""Simulate and plot approx-PC trajectories for a sweep of relinearize_steps values.

For each layer, produces:
  - traj_layer{N}.png  : PCA-projected clamped-phase trajectories, one line per relinearize_steps
  - pdiff_layer{N}.png : ||approx_acts[t] - pcn_acts[t]|| over time, one line per relinearize_steps
"""
import hydra
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flax import nnx
from pathlib import Path
from tqdm import tqdm

from projectlib.dynamics import approx_pc_step, current_pc_vjp, is_converged, pc_step, run
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


def run_pcn(model, x, rclamp, cfg):
    _, history, _ = run(model, pc_step, x,
                        max_timesteps=cfg.dynamics.clamp_timesteps,
                        tolerance=cfg.dynamics.convergence_tolerance,
                        tau=cfg.dynamics.time_constant,
                        learn=False, rclamp=rclamp,
                        return_act_history=True)
    return [np.asarray(h) for h in history]


def run_approx(model, x, rclamp, cfg, relinearize_steps, pbar=None):
    """Run approx-PC, re-linearizing every relinearize_steps steps (None = once).

    Always runs exactly clamp_timesteps total steps for a fair comparison.
    """
    total_timesteps = int(cfg.dynamics.clamp_timesteps)
    step_size = relinearize_steps or total_timesteps
    remaining = total_timesteps
    history = None
    while remaining > 0:
        n = min(step_size, remaining)
        vjps = current_pc_vjp(model, x)
        _, seg, t = run(model, approx_pc_step, x,
                        vjps=vjps, k=cfg.approx_pc.k,
                        min_timesteps=n,
                        max_timesteps=n,
                        tolerance=cfg.dynamics.convergence_tolerance,
                        tau=cfg.dynamics.time_constant,
                        learn=False, rclamp=rclamp,
                        return_act_history=True,
                        disable_tqdm=True)
        history = concatenate_histories(history, seg)
        remaining -= t
        if pbar is not None:
            pbar.update(t)
    return [np.asarray(h) for h in history]


def fit_pca(arrays):
    """Fit PCA on concatenated arrays; return project() fn."""
    combined = np.concatenate(arrays, axis=0)
    mean = combined.mean(axis=0)
    n_dim = combined.shape[1]
    if n_dim == 1:
        def project(t):
            return np.column_stack([np.asarray(t) - mean, np.zeros(len(t))])
    else:
        _, _, vt = np.linalg.svd(combined - mean, full_matrices=False)
        components = vt[:2]
        def project(t):
            return (np.asarray(t) - mean) @ components.T
    return project


def rs_label(rs):
    return "null (full run)" if rs is None else f"{rs} steps"


def relinearization_indices(rs, total_timesteps):
    """Indices into the history array where re-linearization occurred."""
    step_size = rs or total_timesteps
    return list(range(0, total_timesteps, step_size))


@hydra.main(config_path="./configs", config_name="relinearization-sweep", version_base=None)
def main(cfg) -> None:
    out_dir = Path.cwd()
    sns.set_context("paper")
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # Build model
    if cfg.model.activation_fn not in ACTIVATION_FNS:
        raise ValueError(f"Unknown activation_fn: {cfg.model.activation_fn}")
    activation_fn = ACTIVATION_FNS[cfg.model.activation_fn]
    layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
    n_layers = len(layer_dims) - 1
    layer_labels = (
        [f"Hidden 1 (n={layer_dims[1]})"]
        + [f"Hidden {i+2} (n={layer_dims[i+2]})" for i in range(len(layer_dims) - 3)]
        + [f"Output (n={layer_dims[-1]})"]
    )
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    model = MLP(layer_dims, kernel_init_scale=cfg.model.kernel_init_scale,
                activation_fn=activation_fn, rngs=rngs)

    # Free phase
    rng = jrng.PRNGKey(cfg.seed)
    rng, rng_x, rng_n = jrng.split(rng, 3)
    x = jrng.normal(rng_x, (cfg.model.input_dim,))
    _, _, t_free = run(model, pc_step, x,
                       min_timesteps=cfg.dynamics.free_min_timesteps,
                       max_timesteps=cfg.dynamics.free_max_timesteps,
                       tolerance=cfg.dynamics.convergence_tolerance,
                       tau=cfg.dynamics.time_constant,
                       learn=False)
    print(f"Free phase converged in {t_free} steps.")

    fp_acts = model.get_activations()
    graph_def, fp_state = nnx.split(model)

    # Clamp target
    noise_dir = jrng.normal(rng_n, fp_acts[-1].shape)
    noise_dir = noise_dir / jnp.linalg.norm(noise_dir)
    rclamp = fp_acts[-1] + noise_dir * cfg.noise

    # Run PCN
    pcn_model = nnx.merge(graph_def, fp_state)
    pcn_hist = run_pcn(pcn_model, x, rclamp, cfg)
    print("PCN done.")

    # Run approx-PC for each relinearize_steps
    relinearize_steps_sweep = [None if rs is None else int(rs)
                                for rs in cfg.relinearize_steps_sweep]
    traj_plot_vals = [None if rs is None else int(rs)
                      for rs in cfg.get("trajectory_plot_vals", cfg.relinearize_steps_sweep)]
    approx_hists = {}
    for rs in relinearize_steps_sweep:
        with tqdm(total=int(cfg.dynamics.clamp_timesteps),
                  desc=f"approx-PC  rs={rs_label(rs)}") as pbar:
            approx_model = nnx.merge(graph_def, fp_state)
            approx_hists[rs] = run_approx(approx_model, x, rclamp, cfg, rs, pbar=pbar)

    pcn_color = "black"
    palette = sns.color_palette("husl", len(traj_plot_vals))

    # Final loss plot (all sweep values)
    x_labels = [rs_label(rs) for rs in relinearize_steps_sweep]
    fig_loss, ax_loss = plt.subplots(figsize=(max(4, 0.6 * len(relinearize_steps_sweep)), 3.5))
    layer_palette = sns.color_palette("colorblind", n_layers)
    for layer_idx in range(n_layers):
        pcn_final = pcn_hist[layer_idx][-1]
        final_diffs = [
            float(np.linalg.norm(approx_hists[rs][layer_idx][-1] - pcn_final))
            for rs in relinearize_steps_sweep
        ]
        ax_loss.plot(x_labels, final_diffs, marker="o", lw=1.5,
                     color=layer_palette[layer_idx], label=layer_labels[layer_idx])
    ax_loss.set_xlabel("relinearize_steps", fontsize=10)
    ax_loss.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx}^T - \mathbf{v}_\mathrm{PCN}^T\|$", fontsize=10)
    ax_loss.set_title(f"Final activation diff  |  noise={cfg.noise:.2f}", fontsize=11, fontweight="bold")
    ax_loss.legend(fontsize=7, frameon=False, loc="best")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    sns.despine(ax=ax_loss)
    fig_loss.tight_layout()
    out_loss = out_dir / "final_loss.png"
    fig_loss.savefig(out_loss, dpi=cfg.dpi, bbox_inches="tight")
    print(f"Saved {out_loss}")
    plt.close(fig_loss)

    # Per-layer trajectory/pdiff plots (trajectory_plot_vals subset only)
    for layer_idx in range(n_layers):
        pcn_layer = pcn_hist[layer_idx]          # (T_pcn, n_neurons)
        approx_layers = {rs: approx_hists[rs][layer_idx] for rs in traj_plot_vals}
        title = f"{layer_labels[layer_idx]}  |  noise={cfg.noise:.2f}"

        # PCA trajectory overlay
        project = fit_pca([pcn_layer] + list(approx_layers.values()))
        pcn_proj = project(pcn_layer)

        # assign alphas: smaller relinearize_steps → higher alpha; null → red
        finite_rs = sorted([rs for rs in relinearize_steps_sweep if rs is not None])
        n_finite = len(finite_rs)
        alpha_map = {rs: 1.0 - 0.6 * i / max(n_finite - 1, 1)
                     for i, rs in enumerate(finite_rs)}  # smallest rs → alpha 1.0

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(pcn_proj[:, 0], pcn_proj[:, 1], color=pcn_color, lw=2,
                label="PCN", zorder=10)
        ax.scatter(pcn_proj[0, 0], pcn_proj[0, 1], marker="o", s=50,
                   facecolors="none", edgecolors=pcn_color, lw=1.5, zorder=11)
        ax.scatter(pcn_proj[-1, 0], pcn_proj[-1, 1], marker="+", s=100,
                   color=pcn_color, lw=1.5, zorder=11)
        for rs, approx_layer in approx_layers.items():
            proj = project(approx_layer)
            color = "crimson" if rs is None else "steelblue"
            alpha = 1.0 if rs is None else alpha_map[rs]
            ax.plot(proj[:, 0], proj[:, 1], color=color, alpha=alpha, lw=1.5,
                    label=f"approx  rs={rs_label(rs)}")
            ax.scatter(proj[0, 0], proj[0, 1], marker="o", s=30, alpha=alpha,
                       facecolors="none", edgecolors=color, lw=1.2)
            ax.scatter(proj[-1, 0], proj[-1, 1], marker="+", s=80,
                       color=color, alpha=alpha, lw=1.2)
            lin_idx = relinearization_indices(rs, int(cfg.dynamics.clamp_timesteps))
            valid = [i for i in lin_idx if i < len(proj)]
            ax.scatter(proj[valid, 0], proj[valid, 1], marker=".", s=25,
                       color=color, alpha=alpha, zorder=8)
        ax.set_xlabel("PC 1", fontsize=10)
        ax.set_ylabel("PC 2", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, loc="best")
        sns.despine(ax=ax, offset=10)
        fig.tight_layout()
        out = out_dir / "traj" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)

        # pdiff over time
        fig, ax = plt.subplots(figsize=(6, 3))
        for (rs, approx_layer), color in zip(approx_layers.items(), palette):
            min_t = min(len(pcn_layer), len(approx_layer))
            diff = np.linalg.norm(approx_layer[:min_t] - pcn_layer[:min_t], axis=-1)
            ax.plot(diff, color=color, lw=1.5, label=f"rs={rs_label(rs)}")
        ax.set_xlabel("timestep", fontsize=10)
        ax.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx} - \mathbf{v}_\mathrm{PCN}\|$", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7, frameon=False, loc="best")
        sns.despine(ax=ax)
        fig.tight_layout()
        out = out_dir / "pdiff" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)

        # PCA trajectory — side by side
        n_rs = len(traj_plot_vals)
        fig, axes = plt.subplots(1, n_rs, figsize=(4.5 * n_rs, 4.5),
                                 sharex=True, sharey=True)
        if n_rs == 1:
            axes = [axes]
        fig.suptitle(title, fontsize=11, fontweight="bold")
        for ax, (rs, approx_layer) in zip(axes, approx_layers.items()):
            proj = project(approx_layer)
            ax.plot(pcn_proj[:, 0], pcn_proj[:, 1], color=pcn_color, lw=2,
                    label="PCN", zorder=10)
            ax.scatter(pcn_proj[0, 0], pcn_proj[0, 1], marker="o", s=50,
                       facecolors="none", edgecolors=pcn_color, lw=1.5, zorder=11)
            ax.scatter(pcn_proj[-1, 0], pcn_proj[-1, 1], marker="+", s=100,
                       color=pcn_color, lw=1.5, zorder=11)
            ax.plot(proj[:, 0], proj[:, 1], color="steelblue", lw=1.5,
                    label="approx", zorder=9)
            ax.scatter(proj[0, 0], proj[0, 1], marker="o", s=30,
                       facecolors="none", edgecolors="steelblue", lw=1.2)
            ax.scatter(proj[-1, 0], proj[-1, 1], marker="o", s=60,
                       color="darkorange", zorder=12)
            lin_idx = relinearization_indices(rs, int(cfg.dynamics.clamp_timesteps))
            valid = [i for i in lin_idx if i < len(proj)]
            ax.scatter(proj[valid, 0], proj[valid, 1], marker=".", s=30,
                       color="crimson", zorder=13)
            ax.set_title(f"rs={rs_label(rs)}", fontsize=9)
            ax.set_xlabel("PC 1", fontsize=9)
            ax.legend(fontsize=7, frameon=False, loc="best")
            sns.despine(ax=ax, offset=10)
        axes[0].set_ylabel("PC 2", fontsize=9)
        fig.tight_layout()
        out = out_dir / "traj_sidebyside" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
