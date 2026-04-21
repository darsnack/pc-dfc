"""Plot activation trajectories for a single condition from approximation-sweeps results.

Produces per-layer raw timeseries and PCA-space trajectory plots. In PCA space,
re-linearization events and nudge boundaries are marked with distinct scatter points.

Example:
  uv run approximation-sweeps-plot-traj.py experiment_dir=./outputs \\
      noise=1.0 layers=6 rs=null sf=1.0
"""
import json
import pickle
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import to_absolute_path
from pathlib import Path


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


def relinearization_indices(nudge_step_counts: list[int],
                             relinearize_steps,
                             history_len: int) -> list[int]:
    """History indices where re-linearization occurred."""
    indices = []
    cumulative = 0
    for nudge_steps in nudge_step_counts:
        step_size = relinearize_steps if relinearize_steps is not None else nudge_steps
        t = 0
        while t < nudge_steps:
            idx = cumulative + t
            if idx < history_len:
                indices.append(idx)
            t += step_size
        cumulative += nudge_steps
    return indices


def nudge_boundary_indices(nudge_step_counts: list[int], history_len: int) -> list[int]:
    """History indices where each nudge after the first begins."""
    indices = []
    cumulative = 0
    for steps in nudge_step_counts[:-1]:
        cumulative += steps
        if cumulative < history_len:
            indices.append(cumulative)
    return indices


def find_index(sweep: list, value, name: str) -> int:
    for i, v in enumerate(sweep):
        if v == value:
            return i
    raise ValueError(f"{name}={value!r} not found in sweep {sweep}")


@hydra.main(config_path="./configs", config_name="approx-sweep-plot-traj", version_base=None)
def main(cfg) -> None:
    base_dir = Path(to_absolute_path(cfg.experiment_dir)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {base_dir}")

    sns.set_context("paper")
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    manifest_file = base_dir / "approximation_sweeps_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError("Missing manifest. Run approximation-sweeps.py first.")
    with manifest_file.open("r") as f:
        manifest = json.load(f)

    # Find the pkl for the requested layer count
    target_layers = int(cfg.layers)
    run_meta = next((r for r in manifest["runs"] if r["total_layers"] == target_layers), None)
    if run_meta is None:
        available = [r["total_layers"] for r in manifest["runs"]]
        raise ValueError(f"layers={target_layers} not found. Available: {available}")

    with (base_dir / run_meta["data_file"]).open("rb") as f:
        sim_data = pickle.load(f)

    noise_sweep = [float(x) for x in sim_data["noise_sweep"]]
    rs_sweep    = sim_data["relinearize_steps_sweep"]  # list of int or None
    sf_sweep    = [float(x) for x in sim_data["scaling_factor_sweep"]]
    layer_labels = sim_data["layer_labels"]
    fp_output   = np.asarray(sim_data["fp_output"])
    noise_dir   = np.asarray(sim_data["noise_dir"])
    free_history = sim_data["free_history"]
    n_layers    = len(layer_labels)

    # Resolve requested condition
    cfg_rs    = None if cfg.rs is None else int(cfg.rs)
    noise_idx = find_index(noise_sweep, float(cfg.noise), "noise")
    rs_idx    = find_index(rs_sweep, cfg_rs, "rs")
    sf_idx    = find_index(sf_sweep, float(cfg.sf), "sf")
    magnitude = noise_sweep[noise_idx]

    pcn_hist    = sim_data["pcn_sweep_history"][noise_idx]
    approx_hist = sim_data["approx_sweep_history"][rs_idx][sf_idx][noise_idx]
    nudge_steps = sim_data["approx_nudge_steps"][rs_idx][sf_idx][noise_idx]

    rs_str = "null" if cfg_rs is None else str(cfg_rs)
    title_suffix = f"noise={magnitude:.2f},  rs={rs_str},  sf={cfg.sf:.2f},  layers={target_layers}"

    # Full trajectories (free phase + clamped phase)
    t_free = free_history[0].shape[0]  # rows in free phase history
    pcn_full    = [np.concatenate([free_history[l], pcn_hist[l][1:]],    axis=0) for l in range(n_layers)]
    approx_full = [np.concatenate([free_history[l], approx_hist[l][1:]], axis=0) for l in range(n_layers)]

    pcn_color     = sns.color_palette()[0]
    approx_color  = sns.color_palette()[1]
    relin_color   = "crimson"
    nudge_color   = "darkorange"

    out_dir = base_dir / run_meta["out_dir"] / f"traj_noise{magnitude:.2f}_rs{rs_str}_sf{cfg.sf:.2f}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Raw timeseries (one figure per layer) ────────────────────────────────
    for layer_idx in range(n_layers):
        pcn_acts   = pcn_full[layer_idx]
        approx_acts = approx_full[layer_idx]
        n_neurons  = pcn_acts.shape[1]
        T_approx   = approx_hist[layer_idx].shape[0]

        relin_idx = relinearization_indices(nudge_steps, cfg_rs, T_approx)
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

    # ── PCA trajectories (all layers in one figure) ───────────────────────────
    fig_pca, axes_pca = plt.subplots(1, n_layers,
                                     figsize=(4.5 * n_layers, 4.5),
                                     squeeze=False)
    fig_pca.suptitle(f"Clamped-phase PCA  |  {title_suffix}",
                     fontsize=10, fontweight="bold")
    for layer_idx in range(n_layers):
        ax = axes_pca[0, layer_idx]
        pcn_layer    = pcn_hist[layer_idx]
        approx_layer = approx_hist[layer_idx]
        T_approx     = approx_layer.shape[0]

        project, n_components = fit_pca(pcn_layer, approx_layer)
        pcn_proj    = project(pcn_layer)
        approx_proj = project(approx_layer)

        # trajectories
        ax.plot(pcn_proj[:, 0], pcn_proj[:, 1],
                color=pcn_color, lw=1.8, label="PCN")
        ax.plot(approx_proj[:, 0], approx_proj[:, 1],
                color=approx_color, lw=1.8, label="Approx. PC")
        # start / end markers
        ax.scatter(pcn_proj[0, 0], pcn_proj[0, 1],
                   marker="o", s=50, facecolors="none", edgecolors=pcn_color, lw=1.5, zorder=5)
        ax.scatter(pcn_proj[-1, 0], pcn_proj[-1, 1],
                   marker="+", s=100, color=pcn_color, lw=1.5, zorder=5)
        ax.scatter(approx_proj[0, 0], approx_proj[0, 1],
                   marker="o", s=50, facecolors="none", edgecolors=approx_color, lw=1.5, zorder=5)
        ax.scatter(approx_proj[-1, 0], approx_proj[-1, 1],
                   marker="+", s=100, color=approx_color, lw=1.5, zorder=5)
        # re-linearization markers
        relin_idx = relinearization_indices(nudge_steps, cfg_rs, T_approx)
        valid_relin = [i for i in relin_idx if i < len(approx_proj)]
        if valid_relin:
            ax.scatter(approx_proj[valid_relin, 0], approx_proj[valid_relin, 1],
                       marker=".", s=60, color=relin_color, zorder=8, label="Re-linearization")
        # nudge boundary markers
        nudge_idx = nudge_boundary_indices(nudge_steps, T_approx)
        valid_nudge = [i for i in nudge_idx if i < len(approx_proj)]
        if valid_nudge:
            ax.scatter(approx_proj[valid_nudge, 0], approx_proj[valid_nudge, 1],
                       marker="^", s=80, color=nudge_color, zorder=9, label="Nudge boundary")
        ax.set_xlabel("PC 1", fontsize=9)
        ax.set_ylabel("PC 2" if n_components >= 2 else "(1D only)", fontsize=9)
        ax.set_title(layer_labels[layer_idx], fontsize=9)
        ax.set_box_aspect(1)
        ax.legend(fontsize=7, frameon=False, loc="best")
        sns.despine(ax=ax, offset=10)

    fig_pca.tight_layout()
    save_fig(fig_pca, out_dir / "dynamics_pca", cfg.dpi)
    plt.close(fig_pca)


if __name__ == "__main__":
    main()
