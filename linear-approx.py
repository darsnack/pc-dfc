"""
Compare PCN vs approx-PC inference dynamics (no weight learning).

Procedure:
  1. Build a small MLP with random weights.
  2. Run PCN free-phase dynamics until convergence (fixed point).
  3. Sweep over noise magnitudes; for each, clamp the output to
     fp_output + noise_direction * magnitude and run both networks.
  4. Plots (per layer-count)
       dynamics_layer{l}.png  -- per-neuron time series, rows = noise levels
       dynamics_pca.png       -- PC1/PC2 trajectories, rows = noise levels
        dynamics_lineplot.png  -- layer error curves, color gradient by noise magnitude
  5. Summary plot (across layer counts)
       summary_heatmap.png    -- noise magnitude x total layers, max error
"""
import hydra
import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from flax import nnx
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from functools import partial
from pathlib import Path

from projectlib.dynamics import approx_pc_step, current_pc_vjp, pc_step, run
from projectlib.models import MLP
from projectlib.utils import noop

def save_fig(fig, fname, **kwargs):
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight", **kwargs)
    fig.savefig(f"{fname}.eps", dpi=300, bbox_inches="tight", **kwargs)

PLOT_FONT_SIZE = 12
plt.rcParams.update({"font.size": PLOT_FONT_SIZE})

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "white_to_green",
    ["#f7fcf5", "#e5f5e0", "#a1d99b", "#31a354", "#006d2c"],
)
HEATMAP_CMAP.set_under("#f7fcf5")


def heatmap_norm(values, floor):
    """Log-scaled norm that keeps exact zeros white and expands small positives."""
    positive_values = np.asarray(values)[np.asarray(values) > 0]
    if positive_values.size == 0:
        vmin = floor
        vmax = 1.0
    else:
        vmin = max(float(positive_values.min()), floor)
        vmax = float(positive_values.max())
        if vmax <= vmin:
            vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def style_heatmap(ax, values, tolerance):
    """Add faint cell borders and mark near-zero entries with an x."""
    n_rows, n_cols = values.shape
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.6, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if values[row_idx, col_idx] <= tolerance:
                ax.text(
                    col_idx,
                    row_idx,
                    "x",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="#7a7a7a",
                    alpha=0.7,
                )


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

def plot_focus_trajectory(ax, proj, color, label):
    """Draw a single trajectory with visible endpoint markers."""
    ax.plot(proj[:, 0], proj[:, 1], color=color, lw=2.0, label=label, zorder=2)
    ax.scatter(
        proj[0, 0],
        proj[0, 1],
        marker="o",
        s=42,
        color=color,
        edgecolors="white",
        linewidths=0.7,
        zorder=5,
    )
    ax.scatter(
        proj[-1, 0],
        proj[-1, 1],
        marker="+",
        s=180,
        color="white",
        linewidths=4.0,
        zorder=6,
    )
    ax.scatter(
        proj[-1, 0],
        proj[-1, 1],
        marker="+",
        s=130,
        color=color,
        linewidths=2.0,
        zorder=7,
    )

def concat(free, clamped):
    """Stitch free-phase and clamped-phase histories (drop duplicate junction)."""
    return jnp.concatenate([free, clamped[1:]], axis=0)


def run_single_simulation(layer_dims, cfg, out_dir, pca_global_limits=None):
    """Run a single PCN vs approx-PC comparison for a given architecture.

    Saves per-layer dynamics, PCA, and heatmap plots into *out_dir*.
    Returns the (n_noise, n_layers) diff_matrix of final activation errors.
    If pca_global_limits is None, also returns PCA projection data for deferred plotting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    convergence_tol = cfg.dynamics.convergence_tolerance

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
            error_val = np.linalg.norm(
                sweep_pcn_clamp[-1][layer_idx][-1]
                - sweep_approx_clamp[-1][layer_idx][-1]
            )
            diff_matrix[noise_idx, layer_idx] = max(error_val, convergence_tol)

        print(
            f"  noise={magnitude:.2f}  |  layer diffs: "
            + "  ".join(
                f"L{layer_idx + 1}={diff_matrix[noise_idx, layer_idx]:.4f}"
                for layer_idx in range(n_layers)
            )
        )

    # --- per-layer dynamics plots ---
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
                sns.lineplot(
                    x=t_axis,
                    y=pcn_acts[:, neuron_idx],
                    color="steelblue",
                    lw=1.2,
                    label="PCN" if noise_idx == 0 and neuron_idx == n_neurons - 1 else None,
                    ax=ax,
                )
                sns.lineplot(
                    x=t_axis,
                    y=approx_acts[:, neuron_idx],
                    color="darkorange",
                    lw=1.2,
                    linestyle="--",
                    label="approx-PC" if noise_idx == 0 and neuron_idx == n_neurons - 1 else None,
                    ax=ax,
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

        sns.despine(fig)
        plt.tight_layout()
        out = out_dir / f"dynamics_layer{layer_idx + 1}"
        save_fig(fig, out)
        print(f"Saved {out}")
        plt.close(fig)

    # --- PCA trajectory plot ---
    pca_fns = [
        fit_pca(sweep_pcn_clamp[0][layer_idx], sweep_approx_clamp[0][layer_idx])
        for layer_idx in range(n_layers)
    ]

    all_pts_global = []
    for layer_idx in range(n_layers):
        project, _ = pca_fns[layer_idx]
        all_pts_global.extend(
            project(sweep_pcn_clamp[noise_idx][layer_idx])[:, :2]
            for noise_idx in range(n_noise)
        )
        all_pts_global.extend(
            project(sweep_approx_clamp[noise_idx][layer_idx])[:, :2]
            for noise_idx in range(n_noise)
        )
    all_pts = np.concatenate(all_pts_global, axis=0)
    pad = (all_pts.max() - all_pts.min()) * 0.05 + 1e-6
    x_min, x_max = all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad
    y_min, y_max = all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    global_xlim = (x_center - max_range / 2, x_center + max_range / 2)
    global_ylim = (y_center - max_range / 2, y_center + max_range / 2)

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

            sns.lineplot(
                x=pcn_proj[:, 0],
                y=pcn_proj[:, 1],
                color="steelblue",
                lw=1.8,
                sort=False,
                label="PCN" if noise_idx == 0 and layer_idx == n_layers - 1 else None,
                ax=ax,
            )
            sns.lineplot(
                x=approx_proj[:, 0],
                y=approx_proj[:, 1],
                color="darkorange",
                lw=1.8,
                linestyle="--",
                sort=False,
                label="approx-PC" if noise_idx == 0 and layer_idx == n_layers - 1 else None,
                ax=ax,
            )

            ax.scatter(
                pcn_proj[0, 0], pcn_proj[0, 1],
                marker="o", s=50, facecolors="none", edgecolors="steelblue", lw=1.5, zorder=5,
            )
            ax.scatter(
                pcn_proj[-1, 0], pcn_proj[-1, 1],
                marker="+", s=100, color="steelblue", lw=1.5, zorder=5,
            )
            ax.scatter(
                approx_proj[0, 0], approx_proj[0, 1],
                marker="o", s=50, facecolors="none", edgecolors="darkorange", lw=1.5, zorder=5,
            )
            ax.scatter(
                approx_proj[-1, 0], approx_proj[-1, 1],
                marker="+", s=100, color="darkorange", lw=1.5, zorder=5,
            )

            ax.set_xlim(global_xlim)
            ax.set_ylim(global_ylim)
            ax.set_box_aspect(1)
            ax.set_xlabel("PC 1", fontsize=7)
            if noise_idx == 0:
                ax.set_title(layer_labels[layer_idx], fontsize=9)
            if layer_idx == 0:
                ax.set_ylabel(f"noise={magnitude:.2f}\nPC 2", fontsize=7)
            else:
                ax.set_ylabel("PC 2" if n_components >= 2 else "(1-D)", fontsize=7)
            if noise_idx == 0 and layer_idx == n_layers - 1:
                ax.legend(fontsize=7)

    sns.despine(fig_pca, offset=10)
    plt.tight_layout()
    out_pca = out_dir / "dynamics_pca"
    save_fig(fig_pca, out_pca)
    print(f"Saved {out_pca}")
    plt.close(fig_pca)

    # --- standalone PCA focus plot: last hidden layer, largest perturbation ---
    focus_layer_idx = n_layers - 2
    focus_noise_idx = n_noise - 1
    focus_project, focus_n_components = pca_fns[focus_layer_idx]
    focus_pcn_proj = focus_project(sweep_pcn_clamp[focus_noise_idx][focus_layer_idx])[:, :2]
    focus_approx_proj = focus_project(sweep_approx_clamp[focus_noise_idx][focus_layer_idx])[:, :2]

    fig_focus, ax_focus = plt.subplots(figsize=(5.5, 5.0))
    plot_focus_trajectory(ax_focus, focus_pcn_proj, "black", "PCN")
    plot_focus_trajectory(ax_focus, focus_approx_proj, "#c00000", "approx-PC")
    ax_focus.set_xlim(global_xlim)
    ax_focus.set_ylim(global_ylim)
    ax_focus.set_aspect("equal")
    ax_focus.set_box_aspect(1)
    ax_focus.set_xlabel("PC 1", fontsize=10)
    ax_focus.set_ylabel("PC 2" if focus_n_components >= 2 else "(1-D)", fontsize=10)
    ax_focus.set_title(
        f"{layer_labels[focus_layer_idx]}, noise={cfg.experiment.noise_sweep[focus_noise_idx]:.2f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_focus.legend(fontsize=9, loc="best")
    ax_focus.grid(color="#d9d9d9", linewidth=0.6, alpha=0.6)

    out_focus = out_dir / "dynamics_pca_focus.png"
    fig_focus.tight_layout()
    fig_focus.savefig(out_focus, dpi=cfg.dpi)
    print(f"Saved {out_focus}")
    plt.close(fig_focus)

    # --- per-simulation lineplot ---
    fig_lp, ax_lp = plt.subplots(figsize=(max(4, 1.0 * n_layers), 4))

    cmap = plt.cm.Blues
    noise_magnitudes = np.array(cfg.experiment.noise_sweep)
    norm = plt.Normalize(noise_magnitudes.min(), noise_magnitudes.max())

    for noise_idx, magnitude in enumerate(cfg.experiment.noise_sweep):
        color = cmap(0.3 + 0.7 * norm(magnitude))
        ax_lp.plot(
            range(n_layers),
            diff_matrix[noise_idx],
            marker="o",
            markersize=5,
            lw=1.5,
            color=color,
            label=f"{magnitude:.2f}",
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig_lp.colorbar(sm, ax=ax_lp)
    cbar.set_label("Noise magnitude", fontsize=10)
    cbar.outline.set_visible(False)

    ax_lp.set_xticks(range(n_layers))
    ax_lp.set_xticklabels([f"L{layer_idx + 1}" for layer_idx in range(n_layers)], fontsize=9)
    ax_lp.set_xlabel("Layer", fontsize=10)
    ax_lp.set_ylabel("||PCN - approx-PC||", fontsize=10)
    ax_lp.set_title("Final activation error: PCN vs approx-PC", fontsize=11, fontweight="bold", pad=15)

    sns.despine(fig_lp, offset=10)
    plt.tight_layout()
    out_lp = out_dir / "dynamics_lineplot"
    save_fig(fig_lp, out_lp)
    print(f"Saved {out_lp}")
    plt.close(fig_lp)

    return diff_matrix


@hydra.main(config_path="./configs", config_name="linear-approx", version_base=None)
def main(cfg):
    base_dir = Path.cwd()
    noise_sweep = list(cfg.experiment.noise_sweep)
    n_noise = len(noise_sweep)

    # Setup plotting
    sns.set_context("paper")
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    # Determine which layer configurations to run.
    layer_sweep = cfg.experiment.get("layer_sweep", None)

    if layer_sweep is not None:
        total_layers_list = sorted(int(n) for n in layer_sweep)
        hidden_size = int(cfg.model.hidden_size)
    else:
        # Fall back to a single run with the explicit hidden_dims.
        total_layers_list = None

    if total_layers_list is not None:
        # --- sweep over total layer counts ---
        # summary_matrix: rows = noise magnitudes, cols = total layer counts
        summary_matrix = np.zeros((n_noise, len(total_layers_list)))

        for col_idx, total_layers in enumerate(total_layers_list):
            n_hidden = total_layers - 1  # total_layers counts input-to-output edges
            hidden_dims = [hidden_size] * n_hidden
            layer_dims = [cfg.model.input_dim, *hidden_dims, cfg.model.output_dim]
            out_dir = base_dir / f"layers_{total_layers}"

            print(f"\n{'=' * 60}")
            print(f"Running simulation: {total_layers} total layers  "
                  f"(hidden_dims={hidden_dims})")
            print(f"{'=' * 60}")

            diff_matrix = run_single_simulation(layer_dims, cfg, out_dir)
            # Max error across all layers for each noise level.
            summary_matrix[:, col_idx] = diff_matrix.max(axis=1)

        # --- summary heatmap ---
        fig, ax = plt.subplots(
            figsize=(max(3, 1.2 * len(total_layers_list)), max(3, 1.0 * n_noise))
        )

        im = ax.imshow(
            summary_matrix,
            aspect="auto",
            cmap=HEATMAP_CMAP,
             origin="lower",
            norm=heatmap_norm(summary_matrix, cfg.dynamics.convergence_tolerance),
        )
        fig.colorbar(im, ax=ax, label="max ||PCN - approx-PC||")
        ax.set_xticks(range(len(total_layers_list)))
        ax.set_xticklabels([str(n) for n in total_layers_list], fontsize=9)
        ax.set_yticks(range(n_noise))
        ax.set_yticklabels([f"{mag:.2f}" for mag in noise_sweep], fontsize=9)
        ax.set_xlabel("Total layers", fontsize=10)
        ax.set_ylabel("Noise magnitude", fontsize=10)
        ax.set_title(
            "Max activation error across layers: PCN vs approx-PC",
            fontsize=11,
            fontweight="bold",
        )
        style_heatmap(ax, summary_matrix, cfg.dynamics.convergence_tolerance)

        plt.tight_layout()
        out_summary = base_dir / "summary_heatmap.png"
        save_fig(fig, out_summary)
        print(f"\nSaved {out_summary}")
        plt.close(fig)
    else:
        # --- single run (original behaviour) ---
        layer_dims = [cfg.model.input_dim, *cfg.model.hidden_dims, cfg.model.output_dim]
        run_single_simulation(layer_dims, cfg, base_dir)


if __name__ == "__main__":
    main()
