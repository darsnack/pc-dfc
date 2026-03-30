"""
Compare PCN vs approx-PC inference dynamics (no weight learning).

Procedure:
  1. Build a small MLP with random weights.
  2. Run PCN free-phase dynamics until convergence (fixed point).
  3. Sweep over noise magnitudes; for each, clamp the output to
     fp_output + noise_direction * magnitude and run both networks.
  4. Plots
       dynamics_layer{l}.png  -- per-neuron time series, rows = noise levels
       dynamics_pca.png       -- PC1/PC2 trajectories, rows = noise levels
       dynamics_heatmap.png   -- layer x noise heatmap of PCN vs approx-PC error
"""

import hydra
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx
from matplotlib.collections import LineCollection
from functools import partial
from pathlib import Path

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run
from projectlib.models import MLP
from projectlib.utils import noop

def fit_pca(traj_a, traj_b):
    """Fit PCA basis on two trajectories; return a project() fn and n_components."""
    combined = np.concatenate([traj_a, traj_b], axis=0)
    mean = combined.mean(axis=0)
    n_dim = combined.shape[1]
    n_components = min(2, n_dim)

    if n_dim == 1:
        def project(t):
            return np.column_stack([t - mean, np.zeros(len(t))])
    else:
        _, _, v_t = np.linalg.svd(combined - mean, full_matrices=False)
        components = v_t[:n_components]
        def project(t):
            return (np.array(t) - mean) @ components.T

    return project, n_components

def gradient_lc(ax, proj, cmap_name):
    """Draw a time-coloured LineCollection on ax; return handle for legend."""
    segs = np.concatenate([proj[:-1, np.newaxis], proj[1:, np.newaxis]], axis=1)
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(0.35 + 0.65 * np.linspace(0, 1, len(segs)))
    lc = LineCollection(segs, colors=colors, lw=1.8, zorder=2)
    ax.add_collection(lc)
    handle, = ax.plot([], [], color=cmap(0.7), lw=1.8)
    ax.scatter(proj[0, 0], proj[0, 1],
               marker="o",
               s=50,
               color=cmap(0.4),
               zorder=5,
               edgecolors="k",
               lw=0.4)
    ax.scatter(proj[-1, 0], proj[-1, 1],
               marker="+",
               s=100,
               color=cmap(0.9),
               zorder=5,
               lw=1.5)

    return handle

def concat(free, clamped):
    """Stitch free-phase and clamped-phase histories (drop duplicate junction)."""
    return jnp.concatenate([free, clamped[1:]], axis=0)

@hydra.main(config_path="./configs", config_name="linear-approx", version_base=None)
def main(cfg):
    layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
    n_layers = len(layer_dims) - 1
    n_noise = len(cfg.experiment.noise_sweep)

    activation_fns = {
        "identity": noop,
        "noop": noop,
        "tanh": nnx.tanh,
        "relu": nnx.relu,
    }
    if cfg.model.activation_fn not in activation_fns:
        allowed = ", ".join(sorted(activation_fns.keys()))
        raise ValueError(
            f"Unknown activation fn: {cfg.model.activation_fn}. Expected one of: {allowed}."
        )
    activation_fn = activation_fns[cfg.model.activation_fn]

    layer_labels = (
        [f"Hidden 1  (n={layer_dims[1]})"]
        + [f"Hidden {i + 2}  (n={layer_dims[i + 2]})" for i in range(len(layer_dims) - 3)]
        + [f"Output  (n={layer_dims[-1]})"]
    )

    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, activations=cfg.seed + 2)
    model = MLP(layer_dims,
                kernel_init_scale=cfg.model.kernel_init_scale,
                activation_fn=activation_fn,
                rngs=rngs)

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

    tmp_model = nnx.merge(graph_def, fp_state)
    vjps = current_pc_vjp(tmp_model, x)
    approx_step = partial(approx_pc_step, vjps=vjps, k=cfg.approx_pc.k)

    sweep_pcn_clamp = []
    sweep_approx_clamp = []
    sweep_pcn_full = []
    sweep_approx_full = []
    diff_matrix = np.zeros((n_noise, n_layers))

    for noise_idx, magnitude in enumerate(cfg.experiment.noise_sweep):
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
        _, approx_ch, _ = run(
            approx_model,
            approx_step,
            x,
            min_timesteps=cfg.dynamics.clamp_timesteps,
            max_timesteps=cfg.dynamics.clamp_timesteps,
            tolerance=cfg.dynamics.convergence_tolerance,
            tau=cfg.dynamics.time_constant,
            learn=False,
            rclamp=rclamp,
            return_act_history=True,
        )

        sweep_pcn_clamp.append([np.array(h) for h in pcn_ch])
        sweep_approx_clamp.append([np.array(h) for h in approx_ch])
        sweep_pcn_full.append([concat(f, c) for f, c in zip(free_hist, pcn_ch)])
        sweep_approx_full.append([concat(f, c) for f, c in zip(free_hist, approx_ch)])

        for layer_idx in range(n_layers):
            diff_matrix[noise_idx, layer_idx] = np.linalg.norm(
                sweep_pcn_clamp[-1][layer_idx][-1]
                - sweep_approx_clamp[-1][layer_idx][-1]
            )

        print(
            f"  noise={magnitude:.2f}  |  layer diffs: "
            + "  ".join(
                f"L{layer_idx + 1}={diff_matrix[noise_idx, layer_idx]:.4f}"
                for layer_idx in range(n_layers)
            )
        )

    t_axis = np.arange(sweep_pcn_full[0][0].shape[0])
    t_clamp = t_free

    for layer_idx in range(n_layers):
        n_neurons = sweep_pcn_full[0][layer_idx].shape[1]
        fig, axes = plt.subplots(
            n_noise,
            n_neurons,
            figsize=(max(2.8 * n_neurons, 4), 2.5 * n_noise),
            sharex=True,
            sharey="row",
            squeeze=False,
        )
        fig.suptitle(layer_labels[layer_idx], fontsize=13, fontweight="bold")

        for noise_idx, magnitude in enumerate(cfg.experiment.noise_sweep):
            pcn_acts = sweep_pcn_full[noise_idx][layer_idx]
            approx_acts = sweep_approx_full[noise_idx][layer_idx]
            rclamp_val = float(fp_acts[-1][0]) + float(noise_dir[0]) * magnitude

            for neuron_idx in range(n_neurons):
                ax = axes[noise_idx, neuron_idx]
                ax.plot(
                    t_axis,
                    pcn_acts[:, neuron_idx],
                    color="steelblue",
                    lw=1.2,
                    label="PCN",
                )
                ax.plot(
                    t_axis,
                    approx_acts[:, neuron_idx],
                    color="darkorange",
                    lw=1.2,
                    ls="--",
                    label="approx-PC",
                )
                ax.axvline(t_clamp, color="gray", ls=":", lw=0.8)
                if layer_idx == n_layers - 1:
                    ax.axhline(
                        rclamp_val,
                        color="crimson",
                        ls="--",
                        lw=0.9,
                        alpha=0.8,
                    )
                if noise_idx == 0:
                    ax.set_title(f"neuron {neuron_idx}", fontsize=8)
                if neuron_idx == 0:
                    ax.set_ylabel(f"noise\n{magnitude:.2f}", fontsize=7)
                if noise_idx == n_noise - 1:
                    ax.set_xlabel("timestep", fontsize=7)
                if noise_idx == 0 and neuron_idx == n_neurons - 1:
                    ax.legend(fontsize=6, loc="best")

        plt.tight_layout()
        out = Path.cwd() / f"dynamics_layer{layer_idx + 1}.png"
        plt.savefig(out, dpi=cfg.dpi)
        print(f"Saved {out}")
        plt.close(fig)

    pca_fns = [
        fit_pca(sweep_pcn_clamp[0][layer_idx], sweep_approx_clamp[0][layer_idx])
        for layer_idx in range(n_layers)
    ]

    pca_limits = []
    for layer_idx in range(n_layers):
        project, _ = pca_fns[layer_idx]
        all_pts = np.concatenate(
            [
                project(sweep_pcn_clamp[noise_idx][layer_idx])[:, :2]
                for noise_idx in range(n_noise)
            ]
            + [
                project(sweep_approx_clamp[noise_idx][layer_idx])[:, :2]
                for noise_idx in range(n_noise)
            ],
            axis=0,
        )
        pad = (all_pts.max() - all_pts.min()) * 0.05 + 1e-6
        pca_limits.append(
            (
                all_pts[:, 0].min() - pad,
                all_pts[:, 0].max() + pad,
                all_pts[:, 1].min() - pad,
                all_pts[:, 1].max() + pad,
            )
        )

    fig_pca, axes_pca = plt.subplots(
        n_noise,
        n_layers,
        figsize=(4.5 * n_layers, 4.0 * n_noise),
        squeeze=False,
    )
    fig_pca.suptitle("Clamped-phase dynamics -- PC1 vs PC2", fontsize=13, fontweight="bold")

    for noise_idx, magnitude in enumerate(cfg.experiment.noise_sweep):
        for layer_idx in range(n_layers):
            ax = axes_pca[noise_idx, layer_idx]
            project, n_components = pca_fns[layer_idx]
            pcn_proj = project(sweep_pcn_clamp[noise_idx][layer_idx])[:, :2]
            approx_proj = project(sweep_approx_clamp[noise_idx][layer_idx])[:, :2]
            h_pcn = gradient_lc(ax, pcn_proj, "Blues")
            h_approx = gradient_lc(ax, approx_proj, "Oranges")
            xmin, xmax, ymin, ymax = pca_limits[layer_idx]
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect("equal")
            ax.set_xlabel("PC 1", fontsize=7)
            if noise_idx == 0:
                ax.set_title(layer_labels[layer_idx], fontsize=9)
            if layer_idx == 0:
                ax.set_ylabel(f"noise={magnitude:.2f}\nPC 2", fontsize=7)
            else:
                ax.set_ylabel("PC 2" if n_components >= 2 else "(1-D)", fontsize=7)
            if noise_idx == 0 and layer_idx == n_layers - 1:
                ax.legend([h_pcn, h_approx], ["PCN", "approx-PC"], fontsize=7)

    plt.tight_layout()
    out_pca = Path.cwd() / "dynamics_pca.png"
    fig_pca.savefig(out_pca, dpi=cfg.dpi)
    print(f"Saved {out_pca}")
    plt.close(fig_pca)

    fig_hm, ax_hm = plt.subplots(
        figsize=(max(3, 1.2 * n_layers), max(3, 1.0 * n_noise))
    )

    im = ax_hm.imshow(
        diff_matrix,
        aspect="auto",
        cmap="viridis",
        origin="lower",
    )
    fig_hm.colorbar(im, ax=ax_hm, label="||PCN - approx-PC||")

    ax_hm.set_xticks(range(n_layers))
    ax_hm.set_xticklabels([f"L{layer_idx + 1}" for layer_idx in range(n_layers)], fontsize=9)
    ax_hm.set_yticks(range(n_noise))
    ax_hm.set_yticklabels([f"{mag:.2f}" for mag in cfg.experiment.noise_sweep], fontsize=9)
    ax_hm.set_xlabel("Layer", fontsize=10)
    ax_hm.set_ylabel("Noise magnitude", fontsize=10)
    ax_hm.set_title("Final activation error: PCN vs approx-PC", fontsize=11, fontweight="bold")

    for noise_idx in range(n_noise):
        for layer_idx in range(n_layers):
            value = diff_matrix[noise_idx, layer_idx]
            ax_hm.text(
                layer_idx,
                noise_idx,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > diff_matrix.max() * 0.5 else "black",
            )

    plt.tight_layout()
    out_hm = Path.cwd() / "dynamics_heatmap.png"
    fig_hm.tight_layout()
    fig_hm.savefig(out_hm, dpi=cfg.dpi)
    print(f"Saved {out_hm}")
    plt.close(fig_hm)

if __name__ == "__main__":
    main()
