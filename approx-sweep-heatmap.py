"""Plot a pairwise heatmap from approximation-sweeps.py results.

Specify which two sweep dimensions to use as heatmap axes, and fix the
remaining two at specific values via command-line arguments.
Also writes scaling-factor line plots, one per perturbation size, using the
same fixed relinearize_steps value.

summary_matrix axes (from sweep_summary.npz):
  0: relinearize_steps  (null stored as -1)
  1: scaling_factor
  2: noise
  3: total_layers

Example:
  uv run approximation-sweeps-plotting.py experiment_dir=./outputs \\
      row_axis=noise col_axis=layers rs=null sf=1.0
"""
import json
import pickle
import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import to_absolute_path
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "white_to_green",
    ["#f7fcf5", "#e5f5e0", "#a1d99b", "#31a354", "#006d2c"],
)
HEATMAP_CMAP.set_under("#f7fcf5")


def heatmap_norm(values: np.ndarray, floor: float) -> LogNorm:
    positive = values[values > 0]
    if positive.size == 0:
        return LogNorm(vmin=floor, vmax=1.0)
    vmin = max(float(positive.min()), floor)
    vmax = float(positive.max())
    if vmax <= vmin:
        vmax = vmin * 10.0
    return LogNorm(vmin=vmin, vmax=vmax)


def save_fig(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(f"{path}.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}.png")


def find_index(sweep: list, value, name: str) -> int:
    for i, v in enumerate(sweep):
        if v == value:
            return i
    raise ValueError(f"{name}={value!r} not found in sweep {sweep}")


def find_run(manifest: dict, total_layers: int) -> dict:
    for run in manifest.get("runs", []):
        if int(run["total_layers"]) == total_layers:
            return run
    available = [run["total_layers"] for run in manifest.get("runs", [])]
    raise ValueError(f"layers={total_layers} not found in manifest runs {available}")


def plot_heatmap(matrix: np.ndarray,
                 row_ticks: list[str], col_ticks: list[str],
                 row_label: str, col_label: str,
                 title: str,
                 convergence_tolerance: float,
                 out_path: Path, dpi: int) -> None:
    n_rows, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(3, 1.2 * n_cols), max(3, 1.0 * n_rows)))
    im = ax.imshow(matrix, aspect="auto", cmap=HEATMAP_CMAP, origin="lower",
                   norm=heatmap_norm(matrix, convergence_tolerance))
    fig.colorbar(im, ax=ax, label="max ||PCN - approx-PC||")
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_ticks, fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_ticks, fontsize=9)
    ax.set_xlabel(col_label, fontsize=10)
    ax.set_ylabel(row_label, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="#d0d0d0", linestyle="-", linewidth=0.6, alpha=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    for ri in range(n_rows):
        for ci in range(n_cols):
            if matrix[ri, ci] <= convergence_tolerance:
                ax.text(ci, ri, "x", ha="center", va="center",
                        fontsize=9, color="#7a7a7a", alpha=0.7)
    plt.tight_layout()
    save_fig(fig, out_path, dpi)
    plt.close(fig)


def plot_scaling_factor_lines(sim_data: dict,
                              rs_idx: int,
                              rs_label: str,
                              convergence_tolerance: float,
                              total_layers: int,
                              out_dir: Path,
                              dpi: int) -> None:
    """Plot final error vs scaling factor for each noise magnitude.

    sim_data["diff_matrix"] shape is:
      [relinearize_steps, scaling_factor, noise, network_layer]
    sim_data["approx_nudge_counts"] shape is:
      [relinearize_steps][scaling_factor][noise] → int (total nudge count)
    """
    diff_matrix = np.asarray(sim_data["diff_matrix"])
    approx_nudge_counts = sim_data["approx_nudge_counts"]
    sf_sweep = [float(x) for x in sim_data["scaling_factor_sweep"]]
    noise_sweep = [float(x) for x in sim_data["noise_sweep"]]
    layer_labels = [str(x) for x in sim_data["layer_labels"]]

    out_dir.mkdir(parents=True, exist_ok=True)
    x_labels = [f"{sf:g}" for sf in sf_sweep]
    layer_palette = dict(zip(layer_labels, sns.light_palette("seagreen", len(layer_labels) + 1)[1:]))

    for noise_idx, noise in enumerate(noise_sweep):
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(sf_sweep) + 3.5), 5))
        for layer_idx, layer_label in enumerate(layer_labels):
            final_diffs = diff_matrix[rs_idx, :, noise_idx, layer_idx]
            sns.lineplot(x=x_labels, y=final_diffs, marker="o", linewidth=1.5,
                         sort=False, color=layer_palette[layer_label],
                         label=layer_label, ax=ax)

        ax.axhline(convergence_tolerance, color="#7a7a7a", linewidth=0.9,
                   linestyle="--", alpha=0.65)
        ax.set_yscale("log")
        ax.set_xlabel("Scaling factor")
        ax.set_ylabel(r"$\|\mathbf{v}_\mathrm{approx}^T - \mathbf{v}_\mathrm{PCN}^T\|$")
        ax.set_title(f"Final activation diff  |  layers={total_layers}  |  noise={noise:.2f}  |  rs={rs_label}",
                     pad=20, fontweight="bold")
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.2, 1))
        plt.xticks(rotation=45, ha="right")
        sns.despine(ax=ax)

        ax_nudge = ax.twinx()
        nudge_counts = [approx_nudge_counts[rs_idx][sf_idx][noise_idx]
                        for sf_idx in range(len(sf_sweep))]
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

        noise_tag = f"{noise:g}".replace(".", "p").replace("-", "m")
        out_path = out_dir / f"final_loss_scaling_factor_layers{total_layers}_noise{noise_tag}_rs{rs_label}"
        save_fig(fig, out_path, dpi)
        plt.close(fig)


@hydra.main(config_path="./configs", config_name="approx-sweep-heatmap", version_base=None)
def main(cfg) -> None:
    base_dir = Path(to_absolute_path(cfg.experiment_dir)).resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {base_dir}")

    sns.set_context("paper", font_scale=2)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    manifest_file = base_dir / "approximation_sweeps_manifest.json"
    if not manifest_file.exists():
        raise FileNotFoundError("Missing manifest. Run approximation-sweeps.py first.")
    with manifest_file.open("r") as f:
        manifest = json.load(f)

    summary_rel = manifest.get("summary_file")
    if summary_rel is None:
        raise ValueError("No summary_file in manifest — run approximation-sweeps.py with layer_sweep set.")

    npz = np.load(base_dir / summary_rel)
    # summary_matrix[rs_idx, sf_idx, noise_idx, layer_col_idx]
    summary_matrix    = np.asarray(npz["summary_matrix"])
    total_layers_list = [int(x) for x in npz["total_layers_list"]]
    noise_sweep       = [float(x) for x in npz["noise_sweep"]]
    rs_raw            = [int(x) for x in npz["relinearize_steps_sweep"]]  # -1 = null
    sf_sweep          = [float(x) for x in npz["scaling_factor_sweep"]]

    convergence_tolerance = float(cfg.convergence_tolerance)

    # Axis metadata: name → (matrix_axis_index, sweep_values, tick_labels, axis_label)
    axis_info = {
        "rs":     (0, rs_raw,            ["null" if r == -1 else str(r) for r in rs_raw], "Relinearize steps"),
        "sf":     (1, sf_sweep,          [f"{v:.2f}" for v in sf_sweep],                  "Scaling factor"),
        "noise":  (2, noise_sweep,       [f"{v:.2f}" for v in noise_sweep],               "Noise magnitude"),
        "layers": (3, total_layers_list, [str(v) for v in total_layers_list],             "Total layers"),
    }

    row_name = cfg.row_axis
    col_name = cfg.col_axis
    if row_name == col_name:
        raise ValueError("row_axis and col_axis must be different.")
    if row_name not in axis_info or col_name not in axis_info:
        raise ValueError(f"row_axis and col_axis must each be one of: {list(axis_info)}")

    row_ax, _, row_ticks, row_label = axis_info[row_name]
    col_ax, _, col_ticks, col_label = axis_info[col_name]

    # Resolve fixed indices for the two non-axis dimensions.
    # cfg.rs, cfg.sf, cfg.noise, cfg.layers hold the values to fix at.
    fixed_axes = {}
    fixed_desc_parts = []
    fixed_fname_parts = []
    for name, (ax_idx, sweep, ticks, label) in axis_info.items():
        if name in (row_name, col_name):
            continue
        cfg_val = getattr(cfg, name)
        if name == "rs":
            lookup_val = -1 if cfg_val is None else int(cfg_val)
        elif name == "layers":
            lookup_val = int(cfg_val)
        else:
            lookup_val = float(cfg_val)
        idx = find_index(sweep, lookup_val, name)
        fixed_axes[ax_idx] = idx
        fixed_desc_parts.append(f"{label}={ticks[idx]}")
        fixed_fname_parts.append(f"{name}{ticks[idx]}")

    # Slice summary_matrix at fixed indices (descending axis order so earlier
    # slices don't shift remaining axis indices).
    sliced = summary_matrix
    for ax_idx in sorted(fixed_axes.keys(), reverse=True):
        sliced = np.take(sliced, fixed_axes[ax_idx], axis=ax_idx)

    # After slicing, the two remaining axes are in their original sorted order.
    # If row_ax > col_ax the order is (n_cols, n_rows) — transpose to fix.
    if row_ax > col_ax:
        sliced = sliced.T

    fixed_desc = ",  ".join(fixed_desc_parts)
    title = (f"Max activation error: PCN vs approx-PC\n"
             f"{row_label} × {col_label}  |  {fixed_desc}")

    out_path = base_dir / f"heatmap_{row_name}X{col_name}_{'_'.join(fixed_fname_parts)}"
    plot_heatmap(
        matrix=sliced,
        row_ticks=row_ticks, col_ticks=col_ticks,
        row_label=row_label, col_label=col_label,
        title=title,
        convergence_tolerance=convergence_tolerance,
        out_path=out_path,
        dpi=cfg.dpi,
    )

    lineplot_layers = int(cfg.layers)
    line_run = find_run(manifest, lineplot_layers)
    with (base_dir / line_run["data_file"]).open("rb") as f:
        line_sim_data = pickle.load(f)

    line_rs_val = None if cfg.lineplot_rs is None else int(cfg.lineplot_rs)
    line_rs_sweep = [None if rs is None else int(rs)
                     for rs in line_sim_data["relinearize_steps_sweep"]]
    line_rs_idx = find_index(line_rs_sweep, line_rs_val, "lineplot_rs")
    line_rs_label = "null" if line_rs_val is None else str(line_rs_val)
    plot_scaling_factor_lines(
        sim_data=line_sim_data,
        rs_idx=line_rs_idx,
        rs_label=line_rs_label,
        convergence_tolerance=convergence_tolerance,
        total_layers=lineplot_layers,
        out_dir=base_dir / "final_loss_by_scaling_factor",
        dpi=cfg.dpi,
    )


if __name__ == "__main__":
    main()
