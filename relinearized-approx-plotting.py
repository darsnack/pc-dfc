import json
import pickle
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import to_absolute_path
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, LogNorm

LINEPLOT_CMAP = sns.color_palette("Blues", as_cmap=True)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "white_to_green",
    ["#f7fcf5", "#e5f5e0", "#a1d99b", "#31a354", "#006d2c"],
)
HEATMAP_CMAP.set_under("#f7fcf5")

def concatenate_histories(old: list[np.ndarray] | None, new: list[np.ndarray]) -> list[np.ndarray]:
    if old is None:
        return new
    else:
        return [np.concatenate([old_i, new_i[1:]], axis=0) # drop overlapping timestep
                for old_i, new_i in zip(old, new)]

def save_fig(fig: plt.Figure, fname: Path, **kwargs) -> None:
    fig.savefig(f"{fname}.png", bbox_inches="tight", **kwargs)
    # fig.savefig(f"{fname}.eps", bbox_inches="tight", **kwargs)

def load_simulation_data(file_path: Path) -> dict:
    with file_path.open("rb") as f:
        return pickle.load(f)

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

def get_linearization_indices(sim_data: dict) -> list[int]:
    """Return clamped-phase indices where VJPs are recomputed."""
    clamp_steps = int(np.asarray(sim_data["pcn_sweep_history"][0][0]).shape[0]) - 1
    if clamp_steps <= 0:
        return [0]
    relinearize_steps = sim_data.get("relinearize_steps", None)
    if relinearize_steps is None:
        return [0]
    relinearize_steps = int(relinearize_steps)
    if relinearize_steps <= 0:
        return [0]
    return list(range(0, clamp_steps, relinearize_steps))

def draw_linearize_times(ax: plt.Axes, linearization_ts: list[int], with_label: bool):
    """Draw vertical lines at re-linearization boundaries."""
    for idx, t_lin in enumerate(linearization_ts):
        ax.axvline(t_lin, color="#303030", ls="-.", lw=0.75, alpha=0.6,
                   label="linearize" if with_label and idx == 0 else None)

def draw_linearize_points(ax: plt.Axes, approx_proj: np.ndarray,
                          linearization_indices: list[int], with_label: bool):
    """Draw trajectory markers at approx-PC re-linearization points."""
    valid_indices = [idx for idx in linearization_indices if 0 <= idx < len(approx_proj)]
    if not valid_indices:
        return
    pts = approx_proj[valid_indices]
    ax.scatter(pts[:, 0], pts[:, 1], marker="D", s=28, facecolors="white",
               edgecolors="#7f3c00", linewidths=1.0, alpha=0.95, zorder=8,
               label="linearize" if with_label else None)

def plot_single_run(sim_data: dict, out_dir: Path, dpi: int):
    # make sure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)
    # read experiment data
    layer_labels = sim_data["layer_labels"]
    noise_sweep = list(sim_data["noise_sweep"])
    fp_output = np.asarray(sim_data["fp_output"])
    noise_dir = np.asarray(sim_data["noise_dir"])
    free_history = sim_data["free_history"]
    pcn_history = sim_data["pcn_sweep_history"]
    approx_history = sim_data["approx_sweep_history"]
    pcn_full_history = [concatenate_histories(free_history, hist)
                        for hist in pcn_history]
    approx_full_history = [concatenate_histories(free_history, hist)
                           for hist in approx_history]
    diff_matrix = np.asarray(sim_data["diff_matrix"])
    convergence_tol = float(sim_data.get("convergence_tolerance", 0.0))
    diff_matrix_plot = np.where(diff_matrix <= convergence_tol, 0.0, diff_matrix)
    # get parameters
    t_free = free_history[0].shape[0]
    linearization_indices = get_linearization_indices(sim_data)
    linearization_ts = [t_free + idx for idx in linearization_indices]
    n_noise = len(noise_sweep)
    n_layers = diff_matrix.shape[1]

    pcn_color = sns.color_palette()[0]
    approx_color = sns.color_palette()[1]

    # plot potential timeseries
    for layer_idx in range(n_layers):
        n_neurons = free_history[layer_idx].shape[1]
        fig, axes = plt.subplots(n_noise, n_neurons,
                                 figsize=(max(2.8 * n_neurons, 4), 2.5 * n_noise),
                                 sharex=True,
                                 sharey="row",
                                 squeeze=False)
        fig.suptitle(layer_labels[layer_idx], fontsize=13, fontweight="bold")
        for noise_idx, magnitude in enumerate(noise_sweep):
            pcn_acts = pcn_full_history[noise_idx][layer_idx]
            approx_acts = approx_full_history[noise_idx][layer_idx]
            for neuron_idx in range(n_neurons):
                ax = axes[noise_idx, neuron_idx]
                label = "PC" if (noise_idx == 0) and (neuron_idx == n_neurons - 1) else None
                sns.lineplot(x=np.arange(pcn_acts.shape[0]), y=pcn_acts[:, neuron_idx],
                             color=pcn_color, lw=1.2, label=label, ax=ax)
                label = "Approx. PC" if (noise_idx == 0) and (neuron_idx == n_neurons - 1) else None
                sns.lineplot(x=np.arange(approx_acts.shape[0]), y=approx_acts[:, neuron_idx],
                             color=approx_color, lw=1.2, linestyle="--", label=label, ax=ax)
                ax.axvline(t_free, color="gray", ls=":", lw=0.8)
                draw_linearize_times(ax, linearization_ts,
                                     with_label=(noise_idx == 0 and neuron_idx == n_neurons - 1))
                if layer_idx == n_layers - 1:
                    rclamp = float(fp_output[neuron_idx]) + float(noise_dir[neuron_idx]) * magnitude
                    ax.axhline(rclamp, color="crimson", ls="--", lw=0.9, alpha=0.8)
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
        save_fig(fig, out, dpi=dpi)
        print(f"Saved {out}")
        plt.close(fig)

    # fit pca
    pca_fns = [fit_pca(pcn_history[0][layer_idx], approx_history[0][layer_idx])
               for layer_idx in range(n_layers)]
    # plot projected activity
    fig_pca, axes_pca = plt.subplots(n_noise, n_layers,
                                     figsize=(4.5 * n_layers, 4.0 * n_noise),
                                     sharex=True,
                                     sharey=True,
                                     squeeze=False)
    for noise_idx, magnitude in enumerate(noise_sweep):
        for layer_idx in range(n_layers):
            ax = axes_pca[noise_idx, layer_idx]
            project, n_components = pca_fns[layer_idx]
            pcn_proj = project(pcn_history[noise_idx][layer_idx])[:, :2]
            approx_proj = project(approx_history[noise_idx][layer_idx])[:, :2]
            label = "PC" if noise_idx == 0 and layer_idx == n_layers - 1 else None
            sns.lineplot(x=pcn_proj[:, 0], y=pcn_proj[:, 1],
                         color=pcn_color, lw=1.8, sort=False, label=label, ax=ax)
            label = "Approx. PC" if noise_idx == 0 and layer_idx == n_layers - 1 else None
            sns.lineplot(x=approx_proj[:, 0], y=approx_proj[:, 1],
                         color=approx_color, lw=1.8, linestyle="--", sort=False, label=label, ax=ax)
            ax.scatter(pcn_proj[0, 0], pcn_proj[0, 1],
                       marker="o", s=50, facecolors="none", edgecolors=pcn_color, lw=1.5, zorder=5)
            ax.scatter(pcn_proj[-1, 0], pcn_proj[-1, 1],
                       marker="+", s=100, color=pcn_color, lw=1.5, zorder=5)
            ax.scatter(approx_proj[0, 0], approx_proj[0, 1],
                       marker="o", s=50, facecolors="none", edgecolors=approx_color, lw=1.5, zorder=5)
            ax.scatter(approx_proj[-1, 0], approx_proj[-1, 1],
                       marker="+", s=100, color=approx_color, lw=1.5, zorder=5)
            draw_linearize_points(ax, approx_proj, linearization_indices,
                                  with_label=(noise_idx == 0 and layer_idx == n_layers - 1))
            ax.set_box_aspect(1)
            ax.set_xlabel("PC 1", fontsize=7)
            if noise_idx == 0:
                ax.set_title(layer_labels[layer_idx], fontsize=9)
            if layer_idx == 0:
                ax.set_ylabel(f"noise={magnitude:.2f}\nPC 2", fontsize=7)
            else:
                ax.set_ylabel("PC 2" if n_components >= 2 else "None (1D only)", fontsize=7)
            if noise_idx == 0 and layer_idx == n_layers - 1:
                ax.legend(fontsize=7, frameon=False)
    fig_pca.suptitle("Clamped-phase dynamics -- PC1 vs PC2", fontsize=13, fontweight="bold")
    sns.despine(fig_pca, offset=10)
    fig.tight_layout()
    out_pca = out_dir / "dynamics_pca"
    save_fig(fig_pca, out_pca, dpi=dpi)
    print(f"Saved {out_pca}")
    plt.close(fig_pca)

    focus_layer = int(np.argmax(diff_matrix.max(axis=0)))
    focus_project, _ = pca_fns[focus_layer]
    for noise_idx, magnitude in enumerate(noise_sweep):
        focus_pcn_proj = focus_project(pcn_history[noise_idx][focus_layer])[:, :2]
        focus_approx_proj = focus_project(approx_history[noise_idx][focus_layer])[:, :2]

        fig_focus, ax_focus = plt.subplots(figsize=(5.5, 5.0))
        sns.lineplot(x=focus_pcn_proj[:, 0], y=focus_pcn_proj[:, 1],
                     color=pcn_color, lw=1.8, sort=False, label="PC", ax=ax_focus)
        ax_focus.scatter(focus_pcn_proj[0, 0], focus_pcn_proj[0, 1],
                         marker="o", s=50, facecolors="none", edgecolors=pcn_color, lw=1.5, zorder=5)
        ax_focus.scatter(focus_pcn_proj[-1, 0], focus_pcn_proj[-1, 1],
                         marker="+", s=100, color=pcn_color, lw=1.5, zorder=5)
        sns.lineplot(x=focus_approx_proj[:, 0], y=focus_approx_proj[:, 1],
                     color=approx_color, lw=1.8, sort=False, label="Approx PC", ax=ax_focus)
        ax_focus.scatter(focus_approx_proj[0, 0], focus_approx_proj[0, 1],
                         marker="o", s=50, facecolors="none", edgecolors=approx_color, lw=1.5, zorder=5)
        ax_focus.scatter(focus_approx_proj[-1, 0], focus_approx_proj[-1, 1],
                         marker="+", s=100, color=approx_color, lw=1.5, zorder=5)
        draw_linearize_points(ax_focus, focus_approx_proj, linearization_indices, with_label=True)
        sns.despine(ax=ax_focus, offset=10)
        ax_focus.set_aspect("equal")
        ax_focus.set_box_aspect(1)
        ax_focus.set_xlabel("PC 1", fontsize=10)
        ax_focus.set_ylabel("PC 2", fontsize=10)
        ax_focus.set_title(f"{layer_labels[focus_layer]}, noise={magnitude:.2f}",
                           fontsize=11, fontweight="bold")
        ax_focus.legend(fontsize=9, loc="best", frameon=False)

        out_focus = out_dir / f"dynamics_pca_focus_noise{magnitude:.2f}"
        fig_focus.tight_layout()
        save_fig(fig_focus, out_focus, dpi=dpi)
        print(f"Saved {out_focus}")
        plt.close(fig_focus)

    fig_lp, ax_lp = plt.subplots(figsize=(max(4, 1.0 * n_layers), 2))
    norm = plt.Normalize(min(noise_sweep), max(noise_sweep))
    for noise_idx, magnitude in enumerate(noise_sweep):
        color = LINEPLOT_CMAP(0.3 + 0.7 * norm(magnitude))
        sns.lineplot(x=list(range(n_layers)), y=diff_matrix_plot[noise_idx],
                     marker="o", markersize=5, lw=1.5, color=color, label=f"{magnitude:.2f}")
    sns.despine(fig_lp, offset=10)
    ax_lp.legend(frameon=False)

    sm = plt.cm.ScalarMappable(cmap=LINEPLOT_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig_lp.colorbar(sm, ax=ax_lp)
    cbar.set_label("Noise magnitude", fontsize=10)
    ax_lp.set_xticks(range(n_layers))
    ax_lp.set_xticklabels([f"L{layer_idx + 1}" for layer_idx in range(n_layers)], fontsize=9)
    ax_lp.set_xlabel("Layer", fontsize=10)
    ax_lp.set_ylabel("||PCN - approx-PC||", fontsize=10)
    ax_lp.set_title("Final activation error: PCN vs approx-PC",
                    fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    out_lp = out_dir / "dynamics_lineplot"
    save_fig(fig_lp, out_lp, dpi=dpi)
    print(f"Saved {out_lp}")
    plt.close(fig_lp)

def plot_summary_heatmap(summary_matrix: np.ndarray,
                         total_layers_list: list[int],
                         noise_sweep: list[float],
                         convergence_tolerance: float,
                         out_path: Path,
                         dpi: int):
    fig, ax = plt.subplots(figsize=(max(3, 1.2 * len(total_layers_list)),
                                    max(3, 1.0 * len(noise_sweep))))
    im = ax.imshow(summary_matrix,
                   aspect="auto", cmap=HEATMAP_CMAP, origin="lower",
                   norm=heatmap_norm(summary_matrix, convergence_tolerance))
    fig.colorbar(im, ax=ax, label="max ||PCN - approx-PC||")
    ax.set_xticks(range(len(total_layers_list)))
    ax.set_xticklabels([str(n) for n in total_layers_list], fontsize=9)
    ax.set_yticks(range(len(noise_sweep)))
    ax.set_yticklabels([f"{mag:.2f}" for mag in noise_sweep], fontsize=9)
    ax.set_xlabel("Total layers", fontsize=10)
    ax.set_ylabel("Noise magnitude", fontsize=10)
    ax.set_title("Max activation error across layers: PCN vs approx-PC",
                 fontsize=11, fontweight="bold")
    n_rows, n_cols = summary_matrix.shape
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.6, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(n_rows):
        for col_idx in range(n_cols):
            if summary_matrix[row_idx, col_idx] <= convergence_tolerance:
                ax.text(col_idx, row_idx, "x",
                        ha="center", va="center", fontsize=9, color="#7a7a7a", alpha=0.7)
    plt.tight_layout()
    save_fig(fig, out_path, dpi=dpi)
    print(f"Saved {out_path}")
    plt.close(fig)

@hydra.main(config_path="./configs", config_name="relinearized-approx-plotting", version_base=None)
def main(cfg) -> None:
    base_dir = Path(to_absolute_path(cfg.experiment_dir)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {base_dir}")
    # setup seaborn style
    sns.set_context("paper")
    sns.set_style("ticks")
    sns.set_palette("colorblind")
    # read in experiment manifest
    manifest_file = base_dir / "approximation_experiment_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError("Missing experiment manifest. Run approximation-experiment.py first.")
    with manifest_file.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    # plot individual run plots
    for run_meta in manifest.get("runs", []):
        run_dir = base_dir / run_meta["out_dir"]
        data_file = base_dir / run_meta["data_file"]
        sim_data = load_simulation_data(data_file)
        plot_single_run(sim_data, run_dir, cfg.dpi)
    # plot summary file
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
        plot_summary_heatmap(summary_matrix=summary_matrix,
                             total_layers_list=total_layers_list,
                             noise_sweep=noise_sweep,
                             convergence_tolerance=convergence_tolerance,
                             out_path=base_dir / "summary_heatmap",
                             dpi=cfg.dpi)


if __name__ == "__main__":
    main()
