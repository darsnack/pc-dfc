"""Plot nudging activation trajectories, PCA trajectories, and final errors from saved nudging results."""
import pickle
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import to_absolute_path


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(f"{path}.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}.png")


def fit_pca(traj_a: np.ndarray, traj_b: np.ndarray):
    """Fit PCA on two trajectories; return (project_fn, n_components)."""
    combined = np.concatenate([traj_a, traj_b], axis=0)
    mean = combined.mean(axis=0)
    n_dim = combined.shape[1]
    if n_dim == 1:
        def project(t: np.ndarray) -> np.ndarray:
            return np.column_stack([np.asarray(t) - mean, np.zeros(len(t))])
    else:
        _, _, vt = np.linalg.svd(combined - mean, full_matrices=False)
        components = vt[:2]
        def project(t: np.ndarray) -> np.ndarray:
            return (np.asarray(t) - mean) @ components.T
    return project, min(2, n_dim)


def relinearization_indices(nudge_step_counts: list[int], history_len: int) -> list[int]:
    return [int(i) for i in np.cumsum([0, *nudge_step_counts[:-1]]) if i < history_len]


def nudge_boundary_indices(nudge_step_counts: list[int], history_len: int) -> list[int]:
    """History indices where each nudge after the first begins."""
    indices = []
    cumulative = 0
    for steps in nudge_step_counts[:-1]:
        cumulative += steps
        if cumulative < history_len:
            indices.append(cumulative)
    return indices


@hydra.main(config_path="./configs", config_name="nudging-plotting", version_base=None)
def main(cfg):
    base_dir = Path(to_absolute_path(cfg.experiment_dir)).resolve()
    with (base_dir / "simulation_data.pkl").open("rb") as f:
        sim_data = pickle.load(f)

    sns.set_context("paper", font_scale=2)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    sf_idx = list(sim_data["scaling_factors"]).index(float(cfg.sf))
    layer_labels = sim_data["layer_labels"]
    n_layers = len(layer_labels)
    pcn_hist = sim_data["pcn_history"]
    approx_hist = sim_data["approx_histories"][sf_idx]
    nudge_steps = sim_data["nudge_steps"][sf_idx]
    title_suffix = f"noise=1.00,  rs=null,  sf={cfg.sf:.2f},  layers=6"
    pcn_color = sns.color_palette()[0]
    approx_color = sns.color_palette()[1]
    relin_color = "crimson"
    out_dir = base_dir / f"traj_sf{cfg.sf:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    free_history = sim_data["free_history"]
    fp_output = np.asarray(sim_data["fp_output"])
    noise_dir = np.asarray(sim_data["noise_dir"])
    magnitude = 1.0
    nudge_color = "darkorange"
    t_free = free_history[0].shape[0]
    pcn_full = [np.concatenate([free_history[l], pcn_hist[l][1:]], axis=0) for l in range(n_layers)]
    approx_full = [np.concatenate([free_history[l], approx_hist[l][1:]], axis=0) for l in range(n_layers)]

    # ── Raw timeseries (one figure per layer) ────────────────────────────────
    for layer_idx in range(n_layers):
        pcn_acts   = pcn_full[layer_idx]
        approx_acts = approx_full[layer_idx]
        n_neurons  = pcn_acts.shape[1]
        T_approx   = approx_hist[layer_idx].shape[0]

        relin_idx = relinearization_indices(nudge_steps, T_approx)
        nudge_idx = nudge_boundary_indices(nudge_steps, T_approx)

        fig, axes = plt.subplots(1, n_neurons,
                                 figsize=(max(2.8 * n_neurons, 4), 3.0),
                                 sharex=True, sharey=True, squeeze=False)
        fig.suptitle(f"{layer_labels[layer_idx]}  |  {title_suffix}",
                     fontsize=10, fontweight="bold")
        for neuron_idx in range(n_neurons):
            ax = axes[0, neuron_idx]
            last = neuron_idx == n_neurons - 1
            ax.plot(np.arange(pcn_acts.shape[0]), pcn_acts[:, neuron_idx],
                    color=pcn_color, lw=1.2, label="PCN" if last else None)
            ax.plot(np.arange(approx_acts.shape[0]), approx_acts[:, neuron_idx],
                    color=approx_color, lw=1.2, linestyle="--",
                    label="Approx. PC" if last else None)
            # phase boundary
            ax.axvline(t_free, color="gray", ls=":", lw=0.8)
            # clamp target (output layer only)
            if layer_idx == n_layers - 1:
                rclamp_val = float(fp_output[neuron_idx]) + float(noise_dir[neuron_idx]) * magnitude
                ax.axhline(rclamp_val, color="crimson", ls="--", lw=0.9, alpha=0.8)
            # re-linearization lines (in clamped region)
            for ri in relin_idx:
                ax.axvline(t_free + ri, color=relin_color, ls=":", lw=0.6, alpha=0.55,
                           label="Re-lin." if (last and ri == relin_idx[0]) else None)
            # nudge boundary lines
            for ni in nudge_idx:
                ax.axvline(t_free + ni, color=nudge_color, ls="--", lw=0.8, alpha=0.8,
                           label="Nudge" if (last and ni == nudge_idx[0]) else None)
            ax.set_title(f"neuron {neuron_idx}", fontsize=8)
            ax.set_xlabel("timestep", fontsize=8)
        axes[0, 0].set_ylabel("activation", fontsize=8)
        axes[0, -1].legend(fontsize=6, frameon=False, loc="best")
        sns.despine(fig)
        fig.tight_layout()
        save_fig(fig, out_dir / f"dynamics_layer{layer_idx + 1}", cfg.dpi)
        plt.close(fig)

    fig_pca, axes_pca = plt.subplots(1, n_layers,
                                     figsize=(4.5 * n_layers, 4.5),
                                     squeeze=False)
    fig_pca.suptitle(f"Clamped-phase PCA  |  {title_suffix}", fontweight="bold")
    for layer_idx in range(n_layers):
        ax = axes_pca[0, layer_idx]
        pcn_layer    = pcn_hist[layer_idx]
        approx_layer = approx_hist[layer_idx]
        T_approx     = approx_layer.shape[0]

        project, n_components = fit_pca(pcn_layer, approx_layer)
        pcn_proj    = project(pcn_layer)
        approx_proj = project(approx_layer)

        ax.plot(pcn_proj[:, 0], pcn_proj[:, 1],
                color=pcn_color, lw=1.8, label="PCN")
        ax.plot(approx_proj[:, 0], approx_proj[:, 1],
                color=approx_color, lw=1.8, label="Approx. PC")
        ax.scatter(pcn_proj[0, 0], pcn_proj[0, 1],
                   marker="o", s=50, facecolors="none", edgecolors=pcn_color, lw=1.5, zorder=5)
        ax.scatter(pcn_proj[-1, 0], pcn_proj[-1, 1],
                   marker="+", s=100, color=pcn_color, lw=1.5, zorder=5)
        ax.scatter(approx_proj[0, 0], approx_proj[0, 1],
                   marker="o", s=50, facecolors="none", edgecolors=approx_color, lw=1.5, zorder=5)
        ax.scatter(approx_proj[-1, 0], approx_proj[-1, 1],
                   marker="+", s=100, color=approx_color, lw=1.5, zorder=5)
        valid_relin = relinearization_indices(nudge_steps, T_approx)
        if valid_relin:
            ax.scatter(approx_proj[valid_relin, 0], approx_proj[valid_relin, 1],
                       marker=".", s=60, color=relin_color, zorder=8, label="Relinearization")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2" if n_components >= 2 else "(1D only)")
        ax.set_title(layer_labels[layer_idx])
        ax.set_box_aspect(1)
        sns.despine(ax=ax, offset=10)

    handles, labels = [], []
    for ax in axes_pca.flat:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labels:
                handles.append(hi)
                labels.append(li)
    fig_pca.legend(handles, labels, frameon=False, loc="lower center",
                   ncol=len(labels), bbox_to_anchor=(0.5, -0.02))
    fig_pca.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig(fig_pca, out_dir / "dynamics_pca", cfg.dpi)
    plt.close(fig_pca)

    final_errors = np.asarray(sim_data["final_errors"])
    approx_nudge_counts = [len(steps) for steps in sim_data["nudge_steps"]]
    sf_sweep = [float(x) for x in sim_data["scaling_factors"]]

    x_labels = [f"{sf:g}" for sf in sf_sweep]
    layer_palette = dict(zip(layer_labels, sns.light_palette("seagreen", len(layer_labels) + 1)[1:]))

    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(sf_sweep) + 3.5), 5))
    for layer_idx, layer_label in enumerate(layer_labels):
        final_diffs = final_errors[:, layer_idx]
        sns.lineplot(x=x_labels, y=final_diffs, marker="o", linewidth=1.5,
                     sort=False, color=layer_palette[layer_label],
                     label=layer_label, ax=ax)

    ax.axhline(sim_data["nudge_convergence_tolerance"], color="#7a7a7a", linewidth=0.9,
               linestyle="--", alpha=0.65)
    ax.set_yscale("log")
    ax.set_xlabel("Scaling factor")
    ax.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx}^T - \mathbf{v}_\mathrm{PCN}^T\|$")
    ax.set_title("Final activation diff  |  layers=6  |  noise=1.00  |  rs=null",
                 pad=20, fontweight="bold")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.2, 1))
    plt.xticks(rotation=45, ha="right")
    sns.despine(ax=ax)

    ax_nudge = ax.twinx()
    nudge_counts = approx_nudge_counts
    ax_nudge.plot(x_labels, nudge_counts, marker="s", linestyle="--",
                  color="#7a7a7a", linewidth=1.2, markersize=5, zorder=5)
    ax_nudge.set_ylim(0.9, None)
    ax_nudge.set_yscale("log")
    ax_nudge.set_ylabel("# of Nudges")
    nudge_handle = plt.Line2D([], [], color="#7a7a7a", marker="s", linestyle="--",
                               linewidth=1.2, markersize=5, label="Nudges")
    ax_nudge.legend(handles=[nudge_handle], frameon=False, loc="lower left",
                    bbox_to_anchor=(1.2, 0))
    plt.xticks(rotation=45, ha="right")

    fig.tight_layout()

    out_path = base_dir / "final_loss"
    save_fig(fig, out_path, cfg.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
