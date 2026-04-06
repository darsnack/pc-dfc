"""Load approximation experiment artifacts and generate plots."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, LogNorm


def save_fig(fig: plt.Figure, fname: Path, **kwargs) -> None:
    fig.savefig(f"{fname}.png", dpi=300, bbox_inches="tight", **kwargs)
    fig.savefig(f"{fname}.eps", dpi=300, bbox_inches="tight", **kwargs)


PLOT_FONT_SIZE = 12
plt.rcParams.update({"font.size": PLOT_FONT_SIZE})

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "white_to_green",
    ["#f7fcf5", "#e5f5e0", "#a1d99b", "#31a354", "#006d2c"],
)
HEATMAP_CMAP.set_under("#f7fcf5")


def heatmap_norm(values: np.ndarray, floor: float) -> LogNorm:
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


def style_heatmap(ax: plt.Axes, values: np.ndarray, tolerance: float) -> None:
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


def fit_pca(traj_a: np.ndarray, traj_b: np.ndarray):
    """Fit PCA basis on two trajectories; return a project() fn and n_components."""
    combined = np.concatenate([traj_a, traj_b], axis=0)
    mean = combined.mean(axis=0)
    n_dim = combined.shape[1]
    n_components = min(2, n_dim)

    if n_dim == 1:

        def project(t: np.ndarray) -> np.ndarray:
            return np.column_stack([t - mean, np.zeros(len(t))])

    else:
        _, _, v_t = np.linalg.svd(combined - mean, full_matrices=False)
        components = v_t[:n_components]

        def project(t: np.ndarray) -> np.ndarray:
            return (np.array(t) - mean) @ components.T

    return project, n_components


def plot_focus_trajectory(ax: plt.Axes, proj: np.ndarray, color: str, label: str) -> None:
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


def load_simulation_data(file_path: Path) -> dict:
    with file_path.open("rb") as f:
        return pickle.load(f)


def plot_single_run(sim_data: dict, out_dir: Path, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_labels = sim_data["layer_labels"]
    noise_sweep = list(sim_data["noise_sweep"])
    fp_output = np.asarray(sim_data["fp_output"])
    noise_dir = np.asarray(sim_data["noise_dir"])
    sweep_pcn_clamp = sim_data["sweep_pcn_clamp"]
    sweep_approx_clamp = sim_data["sweep_approx_clamp"]
    sweep_pcn_full = sim_data["sweep_pcn_full"]
    sweep_approx_full = sim_data["sweep_approx_full"]
    diff_matrix = np.asarray(sim_data["diff_matrix"])
    t_free = int(sim_data["free_timesteps"])

    n_noise = len(noise_sweep)
    n_layers = diff_matrix.shape[1]

    t_axis = np.arange(sweep_pcn_full[0][0].shape[0])

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

        for noise_idx, magnitude in enumerate(noise_sweep):
            pcn_acts = sweep_pcn_full[noise_idx][layer_idx]
            approx_acts = sweep_approx_full[noise_idx][layer_idx]
            rclamp_val = float(fp_output[0]) + float(noise_dir[0]) * magnitude

            for neuron_idx in range(n_neurons):
                ax = axes[noise_idx, neuron_idx]
                sns.lineplot(
                    x=t_axis,
                    y=pcn_acts[:, neuron_idx],
                    color="steelblue",
                    lw=1.2,
                    label="PCN"
                    if noise_idx == 0 and neuron_idx == n_neurons - 1
                    else None,
                    ax=ax,
                )
                sns.lineplot(
                    x=t_axis,
                    y=approx_acts[:, neuron_idx],
                    color="darkorange",
                    lw=1.2,
                    linestyle="--",
                    label="approx-PC"
                    if noise_idx == 0 and neuron_idx == n_neurons - 1
                    else None,
                    ax=ax,
                )
                ax.axvline(t_free, color="gray", ls=":", lw=0.8)
                if layer_idx == n_layers - 1:
                    ax.axhline(rclamp_val, color="crimson", ls="--", lw=0.9, alpha=0.8)
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

    pca_fns = [
        fit_pca(sweep_pcn_clamp[0][layer_idx], sweep_approx_clamp[0][layer_idx])
        for layer_idx in range(n_layers)
    ]

    all_pts_global = []
    for layer_idx in range(n_layers):
        project, _ = pca_fns[layer_idx]
        all_pts_global.extend(
            project(sweep_pcn_clamp[noise_idx][layer_idx])[:, :2] for noise_idx in range(n_noise)
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

    for noise_idx, magnitude in enumerate(noise_sweep):
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
                pcn_proj[0, 0],
                pcn_proj[0, 1],
                marker="o",
                s=50,
                facecolors="none",
                edgecolors="steelblue",
                lw=1.5,
                zorder=5,
            )
            ax.scatter(
                pcn_proj[-1, 0],
                pcn_proj[-1, 1],
                marker="+",
                s=100,
                color="steelblue",
                lw=1.5,
                zorder=5,
            )
            ax.scatter(
                approx_proj[0, 0],
                approx_proj[0, 1],
                marker="o",
                s=50,
                facecolors="none",
                edgecolors="darkorange",
                lw=1.5,
                zorder=5,
            )
            ax.scatter(
                approx_proj[-1, 0],
                approx_proj[-1, 1],
                marker="+",
                s=100,
                color="darkorange",
                lw=1.5,
                zorder=5,
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
        f"{layer_labels[focus_layer_idx]}, noise={noise_sweep[focus_noise_idx]:.2f}",
        fontsize=11,
        fontweight="bold",
    )
    ax_focus.legend(fontsize=9, loc="best")
    ax_focus.grid(color="#d9d9d9", linewidth=0.6, alpha=0.6)

    out_focus = out_dir / "dynamics_pca_focus.png"
    fig_focus.tight_layout()
    fig_focus.savefig(out_focus, dpi=dpi)
    print(f"Saved {out_focus}")
    plt.close(fig_focus)

    fig_lp, ax_lp = plt.subplots(figsize=(max(4, 1.0 * n_layers), 4))

    cmap = plt.cm.Blues
    noise_magnitudes = np.array(noise_sweep)
    norm = plt.Normalize(noise_magnitudes.min(), noise_magnitudes.max())

    for noise_idx, magnitude in enumerate(noise_sweep):
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
    ax_lp.set_title(
        "Final activation error: PCN vs approx-PC",
        fontsize=11,
        fontweight="bold",
        pad=15,
    )

    sns.despine(fig_lp, offset=10)
    plt.tight_layout()
    out_lp = out_dir / "dynamics_lineplot"
    save_fig(fig_lp, out_lp)
    print(f"Saved {out_lp}")
    plt.close(fig_lp)


def plot_summary_heatmap(
    summary_matrix: np.ndarray,
    total_layers_list: list[int],
    noise_sweep: list[float],
    convergence_tolerance: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(
        figsize=(max(3, 1.2 * len(total_layers_list)), max(3, 1.0 * len(noise_sweep)))
    )

    im = ax.imshow(
        summary_matrix,
        aspect="auto",
        cmap=HEATMAP_CMAP,
        origin="lower",
        norm=heatmap_norm(summary_matrix, convergence_tolerance),
    )
    fig.colorbar(im, ax=ax, label="max ||PCN - approx-PC||")
    ax.set_xticks(range(len(total_layers_list)))
    ax.set_xticklabels([str(n) for n in total_layers_list], fontsize=9)
    ax.set_yticks(range(len(noise_sweep)))
    ax.set_yticklabels([f"{mag:.2f}" for mag in noise_sweep], fontsize=9)
    ax.set_xlabel("Total layers", fontsize=10)
    ax.set_ylabel("Noise magnitude", fontsize=10)
    ax.set_title(
        "Max activation error across layers: PCN vs approx-PC",
        fontsize=11,
        fontweight="bold",
    )
    style_heatmap(ax, summary_matrix, convergence_tolerance)

    plt.tight_layout()
    save_fig(fig, out_path)
    print(f"Saved {out_path}")
    plt.close(fig)


@hydra.main(
    config_path="./configs", config_name="linear-approx-plotting", version_base=None
)
def main(cfg) -> None:
    base_dir = Path(to_absolute_path(cfg.input.experiment_dir)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {base_dir}")

    sns.set_context("paper")
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    manifest_file = base_dir / "approximation_experiment_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError(
            "Missing experiment manifest. Run approximation-experiment.py first."
        )

    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    for run_meta in manifest.get("runs", []):
        run_dir = base_dir / run_meta["out_dir"]
        data_file = base_dir / run_meta["data_file"]
        sim_data = load_simulation_data(data_file)
        plot_single_run(sim_data, run_dir, cfg.dpi)

    summary_rel = manifest.get("summary_file")
    if summary_rel is not None:
        summary_file = base_dir / summary_rel
        summary_npz = np.load(summary_file)
        summary_matrix = np.asarray(summary_npz["summary_matrix"])
        total_layers_list = [int(x) for x in np.asarray(summary_npz["total_layers_list"])]
        noise_sweep = [float(x) for x in np.asarray(summary_npz["noise_sweep"])]
        if "convergence_tolerance" in summary_npz:
            convergence_tolerance = float(np.asarray(summary_npz["convergence_tolerance"]))
        else:
            convergence_tolerance = float(cfg.dynamics.convergence_tolerance)

        plot_summary_heatmap(
            summary_matrix=summary_matrix,
            total_layers_list=total_layers_list,
            noise_sweep=noise_sweep,
            convergence_tolerance=convergence_tolerance,
            out_path=base_dir / "summary_heatmap",
        )


if __name__ == "__main__":
    main()
