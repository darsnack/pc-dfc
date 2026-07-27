"""Plot learning speed comparison across PC, Approx PC, and PC with Nudges.

Downloads per-sample training data and test accuracy from wandb sweeps,
caches it locally, and produces:
  1. Training batch final accuracy vs. integration step (one subplot per sweep value)
  2. Final test accuracy across methods and sweep values

Usage:
  uv run plot-learning-speed.py pc_sweep_id=??? approx_pc_sweep_id=??? pc_nudges_sweep_id=??? \
      [sweep_var=training.learning_rate] [sweep_label='Learning Rate'] [entity=...] [project=...] \
      [cache_dir=...] [no_cache=...]
"""

import pickle
from itertools import product
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from tqdm import tqdm

METHOD_LABELS = {
    "pc": "PC",
    "approx_pc": "Approx PC",
    "pc_nudges": "PC + Nudges",
}
METHOD_KEYS = tuple(METHOD_LABELS)
METHOD_PALETTE = dict(zip(METHOD_LABELS.values(), sns.color_palette("colorblind", n_colors=len(METHOD_LABELS))))


def get_config_value(config, key):
    """Get a W&B config value, supporting both flat and nested dotted keys."""
    if key in config:
        return config[key]

    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def sweep_value_label(value):
    if isinstance(value, float):
        return f"{value:.1e}"
    return str(value)


def sort_sweep_values(values):
    """Sort numeric sweeps numerically while allowing categorical sweep variables."""
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=lambda value: str(value))


def normalize_sweep_config(sweep_var, sweep_label):
    """Normalize one-dimensional or two-dimensional sweep configuration."""
    sweep_vars = [sweep_var] if isinstance(sweep_var, str) else list(sweep_var)
    sweep_labels = [sweep_label] if isinstance(sweep_label, str) else list(sweep_label)
    if len(sweep_vars) not in (1, 2):
        raise ValueError("sweep_var must be a string or a list of exactly two strings.")
    if len(sweep_labels) != len(sweep_vars):
        raise ValueError("sweep_label must have the same number of entries as sweep_var.")
    if not all(isinstance(value, str) for value in sweep_vars + sweep_labels):
        raise ValueError("sweep_var and sweep_label entries must be strings.")
    return sweep_vars, sweep_labels


def run_sweep_value(run_data, sweep_vars):
    return tuple(run_data["sweep_values"][sweep_var] for sweep_var in sweep_vars)


def format_sweep_value(sweep_value, sweep_labels):
    return ", ".join(
        f"{label} = {sweep_value_label(value)}"
        for label, value in zip(sweep_labels, sweep_value)
    )


TRAIN_KEYS = ["train/integration_step", "train/batch_final_accuracy", "train/batch_final_error", "epoch"]


def _is_nan(v):
    if v is None:
        return False
    try:
        return np.isnan(float(v))
    except (ValueError, TypeError):
        return False


def download_run_data(run, cache_dir, no_cache, sweep_vars, pbar=None):
    cache_path = cache_dir / f"{run.id}.pkl"
    if not no_cache and cache_path.exists():
        if pbar is not None:
            pbar.set_postfix_str(f"cached {run.id[:8]}")
        with cache_path.open("rb") as f:
            data = pickle.load(f)
        # Cache config fields separately, so requesting a new sweep variable
        # never requires re-downloading this run's history.
        sweep_values = data.setdefault("sweep_values", {})
        missing_vars = [sweep_var for sweep_var in sweep_vars if sweep_var not in sweep_values]
        if missing_vars:
            for sweep_var in missing_vars:
                sweep_values[sweep_var] = get_config_value(run.config, sweep_var)
            with cache_path.open("wb") as f:
                pickle.dump(data, f)
        if any(sweep_values[sweep_var] is None for sweep_var in sweep_vars):
            missing = [sweep_var for sweep_var in sweep_vars if sweep_values[sweep_var] is None]
            print(f"\n  WARNING: Could not find {missing!r} for run {run.id}")
            return None
        return data

    train_history = []
    for row in run.scan_history(keys=TRAIN_KEYS):
        if "train/integration_step" in row and "train/batch_final_accuracy" in row:
            train_history.append(row)
        if pbar is not None:
            pbar.update(1)

    test_acc = run.summary.get("test/avg_epoch_accuracy")
    test_epoch = run.summary.get("epoch")
    if test_acc is None:
        for row in run.scan_history(keys=["test/avg_epoch_accuracy", "epoch"]):
            if "test/avg_epoch_accuracy" in row:
                test_acc = row["test/avg_epoch_accuracy"]
                test_epoch = row.get("epoch")
            if pbar is not None:
                pbar.update(1)

    sweep_values = {sweep_var: get_config_value(run.config, sweep_var) for sweep_var in sweep_vars}
    if any(value is None for value in sweep_values.values()):
        missing = [sweep_var for sweep_var, value in sweep_values.items() if value is None]
        print(f"\n  WARNING: Could not find {missing!r} for run {run.id}")
        print(f"    Config keys: {list(run.config.keys())}")
        return None

    seed = run.config.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            seed = None

    data = {
        "sweep_values": sweep_values,
        "seed": seed,
        "train_history": train_history,
        "has_nan_loss": any(
            _is_nan(r.get("train/batch_final_error"))
            for r in train_history
        ),
        "test_accuracy": float(test_acc) if test_acc is not None else None,
        "test_epoch": int(test_epoch) if test_epoch is not None else None,
        "run_id": run.id,
        "run_name": run.name,
    }

    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(data, f)

    return data


def collect_sweep_data(api, entity, project, sweep_id, cache_dir, method_key, no_cache, sweep_vars, sweep_labels):
    print(f"\n{METHOD_LABELS[method_key]} (sweep {sweep_id})")
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = list(sweep.runs)
    print(f"  {len(runs)} runs in sweep")
    results = []
    for run_idx, run in enumerate(runs):
        run_values = tuple(get_config_value(run.config, sweep_var) for sweep_var in sweep_vars)
        label = f"run {run_idx + 1}/{len(runs)} {format_sweep_value(run_values, sweep_labels)}"
        pbar = tqdm(desc=label, unit=" rows", leave=True)
        data = download_run_data(run, cache_dir, no_cache, sweep_vars, pbar)
        pbar.close()
        if data is not None:
            data["method"] = method_key
            n_train = len(data["train_history"])
            test_str = f"test_acc={data['test_accuracy']:.3f}" if data["test_accuracy"] is not None else "no test acc"
            seed_str = f"seed={data['seed']}" if data.get("seed") is not None else "no seed"
            print(f"  {run.id[:8]}  {format_sweep_value(run_sweep_value(data, sweep_vars), sweep_labels)}  {seed_str}  train_pts={n_train}  {test_str}")
            results.append(data)
    print(f"  Total: {len(results)}/{len(runs)} runs for {METHOD_LABELS[method_key]}")
    return results


def find_runs_for_sweep_value(runs, sweep_vars, sweep_value, exclude_nan=True):
    return [
        run_data for run_data in runs
        if run_sweep_value(run_data, sweep_vars) == sweep_value
        and (not exclude_nan or not run_data["has_nan_loss"])
    ]


def _accuracy_vs_integration_step_common(all_data, sweep_vars, sweep_values):
    n_values = len(sweep_values)
    if n_values == 0:
        print(f"WARNING: No values found for {sweep_vars}, skipping accuracy vs integration step plot.")
        return None

    n_cols = min(4, n_values)
    n_rows = (n_values + n_cols - 1) // n_cols

    method_order = [METHOD_LABELS[key] for key in METHOD_KEYS if key in all_data]
    palette = METHOD_PALETTE

    all_records = []
    nan_methods_by_value = {}
    for sweep_value in sweep_values:
        nan_methods_by_value[sweep_value] = []
        for method_key in METHOD_KEYS:
            if method_key not in all_data:
                continue
            runs = find_runs_for_sweep_value(all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=False)
            nan_runs = [r for r in runs if r["has_nan_loss"]]
            valid_runs = [r for r in runs if not r["has_nan_loss"]]
            if nan_runs and not valid_runs:
                nan_methods_by_value[sweep_value].append(METHOD_LABELS[method_key])
            for r in valid_runs:
                for h in r["train_history"]:
                    all_records.append({
                        "Sweep Value": sweep_value,
                        "Integration Step": h["train/integration_step"],
                        "Accuracy (%)": h["train/batch_final_accuracy"] * 100,
                        "Method": METHOD_LABELS[method_key],
                        "seed": r.get("seed") if r.get("seed") is not None else r["run_id"],
                    })

    df_all = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    return n_values, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_value


def _finalize_accuracy_subplot(fig, axes, n_values, n_cols, n_rows, method_order, palette, title, out_path, dpi):
    for idx in range(n_values, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles, labels = [], []
    for m in method_order:
        handles.append(plt.Line2D([0], [0], color=palette[m]))
        labels.append(m)
    fig.legend(handles, labels, frameon=False, loc="upper center",
               ncol=len(method_order), bbox_to_anchor=(0.5, 1.0))

    fig.suptitle(title, fontweight="bold", y=1.04)
    fig.tight_layout()

    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {out_path}.png")
    plt.close(fig)


def plot_accuracy_vs_integration_step_mean(all_data, sweep_vars, sweep_labels, sweep_values, output_dir, dpi=200):
    result = _accuracy_vs_integration_step_common(all_data, sweep_vars, sweep_values)
    if result is None:
        return
    n_values, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_value = result

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, sweep_value in enumerate(sweep_values):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_value = df_all[df_all["Sweep Value"] == sweep_value]
            if len(df_value) > 0:
                n_seeds = df_value.groupby(["Method", "Integration Step"]).size().groupby("Method").first()
                has_multi = (n_seeds > 1).any()
                sns.lineplot(
                    data=df_value, x="Integration Step", y="Accuracy (%)",
                    hue="Method", hue_order=method_order, palette=palette,
                    ax=ax, legend=False,
                    errorbar="sd" if has_multi else None,
                )
                mean_curves = []
                if not df_value.empty:
                    for m in method_order:
                        df_m = df_value[df_value["Method"] == m]
                        if len(df_m) > 0:
                            grp = df_m.groupby("Integration Step")["Accuracy (%)"]
                            mean_accs = grp.mean().values
                            steps = grp.mean().index.values
                            mean_curves.append((steps, mean_accs))
                if mean_curves:
                    ax.set_xlim(0, _early_segment_xlim(mean_curves))

        for m in nan_methods_by_value.get(sweep_value, []):
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(format_sweep_value(sweep_value, sweep_labels), fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_value.get(sweep_value):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    _finalize_accuracy_subplot(fig, axes, n_values, n_cols, n_rows, method_order, palette,
                               "Training Accuracy vs. Integration Step (mean ± SD)",
                               output_dir / "accuracy_vs_integration_step_mean", dpi)


def plot_accuracy_vs_integration_step_seeds(all_data, sweep_vars, sweep_labels, sweep_values, output_dir, dpi=200):
    result = _accuracy_vs_integration_step_common(all_data, sweep_vars, sweep_values)
    if result is None:
        return
    n_values, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_value = result

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, sweep_value in enumerate(sweep_values):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_value = df_all[df_all["Sweep Value"] == sweep_value]
            if len(df_value) > 0:
                sns.lineplot(
                    data=df_value, x="Integration Step", y="Accuracy (%)",
                    hue="Method", hue_order=method_order, palette=palette,
                    units="seed", estimator=None,
                    ax=ax, legend=False, alpha=0.4,
                )
                mean_curves = []
                if not df_value.empty:
                    for m in method_order:
                        df_m = df_value[df_value["Method"] == m]
                        if len(df_m) > 0:
                            grp = df_m.groupby("Integration Step")["Accuracy (%)"]
                            mean_accs = grp.mean().values
                            steps = grp.mean().index.values
                            mean_curves.append((steps, mean_accs))
                if mean_curves:
                    ax.set_xlim(0, _early_segment_xlim(mean_curves))

        for m in nan_methods_by_value.get(sweep_value, []):
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(format_sweep_value(sweep_value, sweep_labels), fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_value.get(sweep_value):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    _finalize_accuracy_subplot(fig, axes, n_values, n_cols, n_rows, method_order, palette,
                               "Training Accuracy vs. Integration Step (individual seeds)",
                               output_dir / "accuracy_vs_integration_step_seeds", dpi)


def _early_segment_xlim(curves, plateau_frac=0.9):
    earliest = None
    for steps, accs in curves:
        max_acc = accs.max()
        plateau = max_acc * plateau_frac
        above = np.where(accs >= plateau)[0]
        if len(above) > 0:
            step = steps[above[0]]
            if earliest is None or step < earliest:
                earliest = step
    if earliest is not None and earliest > 0:
        return earliest * 3
    max_step = max(s[-1] for s, _ in curves)
    return max_step


def plot_final_test_accuracy_bar(all_data, sweep_vars, sweep_labels, sweep_values, output_dir, dpi=200):
    if not sweep_values:
        print(f"WARNING: No values found for {sweep_vars}, skipping final test accuracy plot.")
        return

    x_values = sort_sweep_values({value[0] for value in sweep_values})
    facet_values = sort_sweep_values({value[1] for value in sweep_values}) if len(sweep_vars) == 2 else [None]
    method_order = [METHOD_LABELS[key] for key in METHOD_KEYS if key in all_data]
    n_cols = min(4, len(facet_values))
    n_rows = (len(facet_values) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(5.33, len(x_values)) * n_cols, 4 * n_rows), squeeze=False)
    legend_ax = None

    for facet_idx, facet_value in enumerate(facet_values):
        ax = axes[facet_idx // n_cols][facet_idx % n_cols]
        records, nan_records = [], []
        for x_value in x_values:
            sweep_value = (x_value,) if facet_value is None else (x_value, facet_value)
            for method_key in METHOD_KEYS:
                if method_key not in all_data:
                    continue
                all_runs = find_runs_for_sweep_value(all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=False)
                valid_runs = find_runs_for_sweep_value(all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=True)
                if any(run["has_nan_loss"] for run in all_runs) and not valid_runs:
                    nan_records.append({"Sweep Value": sweep_value_label(x_value), "Method": METHOD_LABELS[method_key]})
                records.extend(
                    {"Sweep Value": sweep_value_label(x_value), "Method": METHOD_LABELS[method_key],
                     "Accuracy (%)": run["test_accuracy"] * 100}
                    for run in valid_runs if run["test_accuracy"] is not None
                )

        df = pd.DataFrame(records, columns=["Sweep Value", "Method", "Accuracy (%)"])
        df_nan = pd.DataFrame(nan_records, columns=["Sweep Value", "Method"])
        x_order = [sweep_value_label(value) for value in x_values]
        if not df.empty:
            n_seeds = df.groupby(["Sweep Value", "Method"]).size()
            pointplot_dodge = 0.525 if len(method_order) > 1 else False
            stripplot_dodge = 0.3 if len(method_order) > 1 else False
            sns.pointplot(data=df, x="Sweep Value", y="Accuracy (%)", hue="Method", order=x_order,
                          hue_order=method_order, palette=METHOD_PALETTE, ax=ax, errorbar="sd" if n_seeds.max() > 1 else None,
                          dodge=pointplot_dodge, markers="_", linestyle="none", capsize=0.08, markersize=12)
            sns.stripplot(data=df, x="Sweep Value", y="Accuracy (%)", hue="Method", order=x_order,
                          hue_order=method_order, palette=METHOD_PALETTE, ax=ax, dodge=stripplot_dodge, alpha=0.5, size=6, legend=False)
            if legend_ax is None:
                legend_ax = ax
        for boundary in np.arange(len(x_order) - 1) + 0.5:
            ax.axvline(boundary, color="lightgrey", linewidth=0.8, zorder=0)
        for _, row in df_nan.iterrows():
            ax.text(x_order.index(row["Sweep Value"]), 2, r"$\times$", ha="center", va="bottom", color="black")
        if facet_value is not None:
            ax.set_title(f"{sweep_labels[1]} = {sweep_value_label(facet_value)}", fontweight="bold")
        ax.set_xlabel(sweep_labels[0])
        ax.set_ylim(0, 105)
        sns.despine(ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

    for idx in range(len(facet_values), n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)
    handles, labels = legend_ax.get_legend_handles_labels() if legend_ax is not None else ([], [])
    if handles:
        fig.legend(handles[:len(method_order)], labels[:len(method_order)], frameon=False,
                   loc="upper center", ncol=len(method_order), bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    out_path = output_dir / "final_test_accuracy_point"
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {out_path}.png")
    plt.close(fig)


def print_final_test_accuracy_summary(all_data, sweep_vars, sweep_labels, sweep_values):
    """Print the per-method values represented in the final-accuracy plot."""
    print(f"\nFinal test accuracy summary by {', '.join(sweep_labels)}:")
    print(f"  {'Sweep Value':<20} {'Method':<12} {'Seeds':>5}  Mean ± stderr")
    for sweep_value in sweep_values:
        for method_key in METHOD_KEYS:
            if method_key not in all_data:
                continue
            runs = find_runs_for_sweep_value(
                all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=True
            )
            accuracies = [run["test_accuracy"] * 100 for run in runs
                          if run["test_accuracy"] is not None]
            n_seeds = len(accuracies)
            if n_seeds == 0:
                summary = "no final test accuracy"
            else:
                mean = np.mean(accuracies)
                stderr = np.std(accuracies, ddof=1) / np.sqrt(n_seeds) if n_seeds > 1 else 0.0
                summary = f"{mean:.2f}% ± {stderr:.2f}%"
            print(f"  {format_sweep_value(sweep_value, sweep_labels):<40} "
                  f"{METHOD_LABELS[method_key]:<12} {n_seeds:>5}  {summary}")


@hydra.main(config_path="./configs", config_name="plot-learning-speed", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path.cwd()
    cache_dir = Path(get_original_cwd()) / cfg.cache_dir

    sns.set_context("paper", font_scale=2)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    sweep_vars, sweep_labels = normalize_sweep_config(cfg.sweep_var, cfg.sweep_label)

    api = wandb.Api(timeout=60)

    all_data = {}
    for method_key, sweep_id in [
        ("pc", cfg.pc_sweep_id),
        ("approx_pc", cfg.approx_pc_sweep_id),
        ("pc_nudges", cfg.pc_nudges_sweep_id),
    ]:
        if sweep_id is None:
            print(f"\nSkipping {METHOD_LABELS[method_key]}: no sweep ID configured.")
            continue
        all_data[method_key] = collect_sweep_data(
            api, cfg.entity, cfg.project, sweep_id, cache_dir, method_key, cfg.no_cache,
            sweep_vars, sweep_labels
        )

    values_by_var = [sort_sweep_values({run_data["sweep_values"][sweep_var]
                                        for runs in all_data.values() for run_data in runs})
                     for sweep_var in sweep_vars]
    sweep_values = list(product(*values_by_var))
    print(f"\nSweep grid: {[format_sweep_value(value, sweep_labels) for value in sweep_values]}")

    print("\nSample sizes (total / non-NaN / with test acc):")
    for sweep_value in sweep_values:
        for method_key in METHOD_KEYS:
            if method_key not in all_data:
                continue
            all_runs = find_runs_for_sweep_value(all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=False)
            valid_runs = find_runs_for_sweep_value(all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=True)
            with_test = [r for r in valid_runs if r["test_accuracy"] is not None]
            print(f"  {METHOD_LABELS[method_key]:12s} {format_sweep_value(sweep_value, sweep_labels)}  {len(all_runs)} / {len(valid_runs)} / {len(with_test)}")

    plot_accuracy_vs_integration_step_mean(
        all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi
    )
    plot_accuracy_vs_integration_step_seeds(
        all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi
    )
    plot_final_test_accuracy_bar(
        all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi
    )
    print_final_test_accuracy_summary(all_data, sweep_vars, sweep_labels, sweep_values)


if __name__ == "__main__":
    main()
