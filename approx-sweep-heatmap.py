"""Plot a pairwise heatmap from approximation-sweeps.py results.

Specify which two sweep dimensions to use as heatmap axes, and fix the
remaining two at specific values via command-line arguments.

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


@hydra.main(config_path="./configs", config_name="approx-sweep-heatmap", version_base=None)
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


if __name__ == "__main__":
    main()
