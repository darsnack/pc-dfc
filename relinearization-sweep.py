"""Simulate and plot approx-PC trajectories for a sweep of relinearize_steps values.

For each layer, produces:
  - traj_layer{N}.png  : PCA-projected clamped-phase trajectories, one line per relinearize_steps
  - pdiff_layer{N}.png : ||approx_acts[t] - pcn_acts[t]|| over time, one line per relinearize_steps
  - deltaflow_layer{N}.png : ||Δflow_l(t)|| over time along the 'relinearize never' trajectory
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

    Stops early if the system converges before exhausting clamp_timesteps.
    Returns (history, n_relinearizations) where n_relinearizations excludes the
    initial linearization (i.e., counts only subsequent re-linearizations).
    """
    total_timesteps = int(cfg.dynamics.clamp_timesteps)
    step_size = relinearize_steps or total_timesteps
    remaining = total_timesteps
    history = None
    n_vjp = 0
    while remaining > 0:
        n = min(step_size, remaining)
        vjps = current_pc_vjp(model, x)
        n_vjp += 1
        _, seg, t = run(model, approx_pc_step, x,
                        vjps=vjps, k=cfg.approx_pc.k,
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
        if (step_size == total_timesteps) or is_converged(seg, cfg.dynamics.convergence_tolerance, "norm"):
            break
    n_relinearizations = max(0, n_vjp - 1)
    return [np.asarray(h) for h in history], n_relinearizations


def fit_pca(arrays):
    """Fit PCA on concatenated arrays; return (project, mean, components)."""
    combined = np.concatenate(arrays, axis=0)
    mean = combined.mean(axis=0)
    n_dim = combined.shape[1]
    if n_dim == 1:
        components = np.array([[1.0], [0.0]])
        def project(t):
            return np.column_stack([np.asarray(t) - mean, np.zeros(len(t))])
    else:
        _, _, vt = np.linalg.svd(combined - mean, full_matrices=False)
        components = vt[:2]
        def project(t):
            return (np.asarray(t) - mean) @ components.T
    return project, mean, components


def rs_label(rs, break_line=False):
    if rs is None:
        return "relinearize\nnever" if break_line else "relinearize never"
    return f"relinearize every\n{rs} steps" if break_line else f"relinearize every {rs} steps"


def relinearization_indices(rs, total_timesteps):
    """Indices into the history array where re-linearization occurred."""
    step_size = rs or total_timesteps
    return list(range(0, total_timesteps, step_size))


def compute_trajectory_delta_flow(graph_def, fp_state, initial_vjps,
                                  approx_never_hist, x, k, tau, rclamp):
    """Compute ||Δflow|| at each timestep along the 'relinearize never' trajectory.

    At each timestep t, compares one step of approx_pc_step with VJPs
    recomputed at the trajectory state vs. with the initial (frozen) VJPs.
    Returns array of shape (T, n_layers) with ||Δdv_l(t)|| per layer.
    """
    n_layers = len(approx_never_hist)
    T = approx_never_hist[0].shape[0]
    delta_flow = np.zeros((T, n_layers))

    for t in range(T):
        model_local = nnx.merge(graph_def, fp_state)
        for l in range(n_layers):
            model_local.activations[l].value = jnp.asarray(approx_never_hist[l][t])
        if rclamp is not None:
            model_local.activations[-1].value = jnp.asarray(rclamp)
        local_vjps = current_pc_vjp(model_local, x)

        v_before_local = [np.asarray(model_local.activations[l].value) for l in range(n_layers)]
        approx_pc_step(model_local, x, vjps=local_vjps, k=k, tau=tau,
                       rclamp=rclamp, learn=False)
        dv_local = [np.asarray(model_local.activations[l].value) - v_before_local[l]
                    for l in range(n_layers)]

        model_init = nnx.merge(graph_def, fp_state)
        for l in range(n_layers):
            model_init.activations[l].value = jnp.asarray(approx_never_hist[l][t])
        if rclamp is not None:
            model_init.activations[-1].value = jnp.asarray(rclamp)

        v_before_init = [np.asarray(model_init.activations[l].value) for l in range(n_layers)]
        approx_pc_step(model_init, x, vjps=initial_vjps, k=k, tau=tau,
                       rclamp=rclamp, learn=False)
        dv_init = [np.asarray(model_init.activations[l].value) - v_before_init[l]
                   for l in range(n_layers)]

        for l in range(n_layers):
            delta_flow[t, l] = np.linalg.norm(dv_local[l] - dv_init[l])

    return delta_flow


@hydra.main(config_path="./configs", config_name="relinearization-sweep", version_base=None)
def main(cfg) -> None:
    out_dir = Path.cwd()
    sns.set_context("paper", font_scale=2)
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

    initial_vjps = current_pc_vjp(model, x)

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
    loss_plot_vals = [None if rs is None else int(rs)
                      for rs in cfg.get("loss_plot_vals", cfg.relinearize_steps_sweep)]
    approx_hists = {}
    approx_relin_counts = {}
    for rs in relinearize_steps_sweep:
        with tqdm(total=int(cfg.dynamics.clamp_timesteps),
                  desc=f"approx-PC  rs={rs_label(rs)}") as pbar:
            approx_model = nnx.merge(graph_def, fp_state)
            hist, n_relin = run_approx(approx_model, x, rclamp, cfg, rs, pbar=pbar)
            approx_hists[rs] = hist
            approx_relin_counts[rs] = n_relin

    palette = sns.light_palette("seagreen", len(traj_plot_vals) + 1)[1:]

    # Final loss plot helper
    layer_palette = dict(zip(layer_labels, sns.light_palette("seagreen", n_layers + 1)[1:]))

    def make_final_loss_plot(rs_values, filename, figsize):
        x_labels = ["never" if rs is None else str(rs) for rs in rs_values]
        fig_loss, ax_loss = plt.subplots(figsize=figsize)
        for layer_idx in range(n_layers):
            pcn_final = pcn_hist[layer_idx][-1]
            final_diffs = [
                float(np.linalg.norm(approx_hists[rs][layer_idx][-1] - pcn_final))
                for rs in rs_values
            ]
            sns.lineplot(x=x_labels, y=final_diffs, marker="o", linewidth=1.5, sort=False,
                         color=layer_palette[layer_labels[layer_idx]], label=layer_labels[layer_idx],
                         ax=ax_loss)
        ax_loss.set_xlabel("Relinearization Frequency (steps)")
        ax_loss.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx}^T - \mathbf{v}_\mathrm{PCN}^T\|$")
        ax_loss.set_yscale("log")
        ax_loss.set_title(f"Final activation diff  |  noise={cfg.noise:.2f}", pad=20, fontweight="bold")
        ax_loss.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.3, 1))
        plt.xticks(rotation=45, ha="right")
        sns.despine(ax=ax_loss)

        rs_relin = [rs for rs in rs_values if rs is not None]
        relin_x_labels = [str(rs) for rs in rs_relin]
        relin_counts = [approx_relin_counts[rs] for rs in rs_relin]
        ax_relin = ax_loss.twinx()
        ax_relin.plot(relin_x_labels, relin_counts, marker="s", linestyle="--",
                      color="#7a7a7a", linewidth=1.2, markersize=5, zorder=5)
        ax_relin.set_ylim(0.9, None)
        ax_relin.set_yscale("log")
        ax_relin.set_ylabel("# of Relinearizations")
        relin_handle = plt.Line2D([], [], color="#7a7a7a", marker="s", linestyle="--",
                                   linewidth=1.2, markersize=5, label="Re-linearizations")
        ax_relin.legend(handles=[relin_handle], frameon=False, loc="lower left",
                        bbox_to_anchor=(1.3, 0))
        plt.xticks(rotation=45, ha="right")

        fig_loss.tight_layout()
        out_loss = out_dir / filename
        fig_loss.savefig(out_loss, dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out_loss}")
        plt.close(fig_loss)

    make_final_loss_plot(relinearize_steps_sweep, "final_loss.png",
                         figsize=(max(6, 0.7 * len(relinearize_steps_sweep) + 3.5), 5))
    make_final_loss_plot(loss_plot_vals, "final_loss_subset.png",
                         figsize=(max(6, 0.75 * len(loss_plot_vals) + 3.75), 5))

    # Compute Δflow along the 'relinearize never' trajectory
    approx_never_hist = [approx_hists[None][l] for l in range(n_layers)]
    print("Computing trajectory Δflow...")
    delta_flow = compute_trajectory_delta_flow(
        graph_def, fp_state, initial_vjps, approx_never_hist,
        x, k=cfg.approx_pc.k, tau=cfg.dynamics.time_constant, rclamp=rclamp)
    print("Δflow done.")

    # Per-layer trajectory/pdiff/deltaflow plots (trajectory_plot_vals subset only)
    for layer_idx in range(n_layers):
        pcn_layer = pcn_hist[layer_idx]          # (T_pcn, n_neurons)
        approx_layers = {rs: approx_hists[rs][layer_idx] for rs in traj_plot_vals}
        title = f"{layer_labels[layer_idx]}  |  noise={cfg.noise:.2f}"

        # PCA trajectory overlay
        project, pca_mean, pca_components = fit_pca([pcn_layer] + list(approx_layers.values()))
        pcn_proj = project(pcn_layer)

        # assign alphas: smaller relinearize_steps → higher alpha
        finite_rs = sorted([rs for rs in relinearize_steps_sweep if rs is not None])
        n_finite = len(finite_rs)
        alpha_map = {rs: 1.0 - 0.85 * i / max(n_finite - 1, 1)
                     for i, rs in enumerate(finite_rs)}

        traj_palette = sns.color_palette()
        pc_color = traj_palette[0]
        approx_color = traj_palette[1]

        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        sns.lineplot(x=pcn_proj[:, 0], y=pcn_proj[:, 1], color=pc_color, linewidth=2,
                     label="PC", zorder=10, sort=False, ax=ax)
        sns.scatterplot(x=[pcn_proj[0, 0]], y=[pcn_proj[0, 1]], marker="o", s=50,
                        facecolors="none", edgecolors=pc_color, linewidth=1.5, zorder=11, ax=ax)
        sns.scatterplot(x=[pcn_proj[-1, 0]], y=[pcn_proj[-1, 1]], marker="+", s=100,
                        color=pc_color, linewidth=1.5, zorder=11, ax=ax)
        for rs, approx_layer in approx_layers.items():
            proj = project(approx_layer)
            alpha = 1.0 if rs is None else alpha_map[rs]
            sns.lineplot(x=proj[:, 0], y=proj[:, 1], color=approx_color, alpha=alpha, linewidth=1.5,
                         label=f"Approx PC ({rs_label(rs)})", sort=False, ax=ax)
            sns.scatterplot(x=[proj[0, 0]], y=[proj[0, 1]], marker="o", s=30, alpha=alpha,
                            facecolors="none", edgecolors=approx_color, linewidth=1.2, ax=ax)
            sns.scatterplot(x=[proj[-1, 0]], y=[proj[-1, 1]], marker="+", s=100,
                            color=approx_color, alpha=alpha, linewidth=1.5, ax=ax)
            lin_idx = relinearization_indices(rs, int(cfg.dynamics.clamp_timesteps))
            valid = [i for i in lin_idx if i < len(proj)]
            if valid:
                sns.scatterplot(x=proj[valid, 0], y=proj[valid, 1], marker=".", s=25,
                                color="red", alpha=alpha, zorder=8, ax=ax)
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_aspect("equal")
        ax.set_box_aspect(1)
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        sns.despine(ax=ax, offset=10)
        fig.tight_layout()
        out = out_dir / "traj" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)

        # pdiff over time
        fig, ax = plt.subplots(figsize=(8, 3))
        for (rs, approx_layer), color in zip(approx_layers.items(), palette):
            min_t = min(len(pcn_layer), len(approx_layer))
            diff = np.linalg.norm(approx_layer[:min_t] - pcn_layer[:min_t], axis=-1)
            sns.lineplot(x=np.arange(len(diff)), y=diff, color=color, linewidth=1.5,
                         label=rs_label(rs), sort=False, ax=ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx} - \mathbf{v}_\mathrm{PCN}\|$")
        ax.set_title(title, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
        sns.despine(ax=ax)
        fig.tight_layout()
        out = out_dir / "pdiff" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)

        # Δflow over time
        fig, ax = plt.subplots(figsize=(8, 3))
        layer_delta = delta_flow[:, layer_idx]
        sns.lineplot(x=np.arange(len(layer_delta)), y=layer_delta,
                     color=layer_palette[layer_labels[layer_idx]], linewidth=1.5, ax=ax)
        ax.set_xlabel("timestep")
        ax.set_ylabel(r"$\|\Delta\mathrm{flow}(t)\|$")
        ax.set_title(title, fontweight="bold")
        sns.despine(ax=ax)
        fig.tight_layout()
        out = out_dir / "deltaflow" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)

        # PCA trajectory — side by side
        n_rs = len(traj_plot_vals)
        fig, axes = plt.subplots(1, n_rs, figsize=(3 * n_rs, 5),
                                  sharex=True, sharey=True)
        if n_rs == 1:
            axes = [axes]
        fig.suptitle(title, fontweight="bold")
        for ax, (rs, approx_layer) in zip(axes, approx_layers.items()):
            proj = project(approx_layer)
            sns.lineplot(x=pcn_proj[:, 0], y=pcn_proj[:, 1], color=pc_color, linewidth=2,
                         zorder=10, sort=False, ax=ax)
            sns.scatterplot(x=[pcn_proj[0, 0]], y=[pcn_proj[0, 1]], marker="o", s=50,
                            facecolors="none", edgecolors=pc_color, linewidth=1.5, zorder=11, ax=ax)
            sns.scatterplot(x=[pcn_proj[-1, 0]], y=[pcn_proj[-1, 1]], marker="+", s=100,
                            color=pc_color, linewidth=1.5, zorder=11, ax=ax)
            sns.lineplot(x=proj[:, 0], y=proj[:, 1], color=approx_color, linewidth=1.5,
                         zorder=9, sort=False, ax=ax)
            sns.scatterplot(x=[proj[0, 0]], y=[proj[0, 1]], marker="o", s=30,
                            facecolors="none", edgecolors=approx_color, linewidth=1.2, ax=ax)
            sns.scatterplot(x=[proj[-1, 0]], y=[proj[-1, 1]], marker="+", s=100,
                            color=approx_color, linewidth=1.5, zorder=12, ax=ax)
            lin_idx = relinearization_indices(rs, int(cfg.dynamics.clamp_timesteps))
            valid = [i for i in lin_idx if i < len(proj)]
            if valid:
                sns.scatterplot(x=proj[valid, 0], y=proj[valid, 1], marker=".", s=30,
                                color="red", zorder=13, ax=ax)
            ax.set_title(rs_label(rs, break_line=True))
            ax.set_xlabel("PC 1")
            ax.set_box_aspect(1)
            sns.despine(ax=ax, offset=10)
        axes[0].set_ylabel("PC 2")
        pc_handle = plt.Line2D([], [], color=pc_color, linewidth=2, label="PC")
        approx_handle = plt.Line2D([], [], color=approx_color, linewidth=1.5, label="Approx PC")
        relin_handle = plt.Line2D([], [], color="red", marker=".", linestyle="None", markersize=6,
                                   label="Relinearization")
        fig.legend(handles=[pc_handle, approx_handle, relin_handle],
                   frameon=False, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
        out = out_dir / "traj_sidebyside" / f"layer{layer_idx + 1}"
        out.parent.mkdir(exist_ok=True)
        fig.savefig(f"{out}.png", dpi=cfg.dpi, bbox_inches="tight")
        print(f"Saved {out}.png")
        plt.close(fig)


if __name__ == "__main__":
    main()
