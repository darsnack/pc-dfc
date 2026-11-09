"""Plot learning speed comparison across PC, Approx PC, and PC with Nudges.

Downloads per-sample training data and test accuracy from wandb sweeps,
caches it locally, and produces:
  1. Training batch final accuracy vs. integration step (one subplot per sweep value)
  2. Training batch final accuracy vs. sample count (matching integration-step plots)
  3. Final test accuracy across methods and sweep values

Usage:
  uv run plot-learning-speed.py pc_sweep_id=??? approx_pc_sweep_id=??? pc_nudges_sweep_id=??? \
      [sweep_var=training.learning_rate] [sweep_label='Learning Rate'] [entity=...] [project=...] \
      [fast=true] [download_workers=4] [history_page_size=10000] [cache_dir=...] [no_cache=...] [bp=88.83] [dfc=88.83] [strong-dfc=88.83]
"""

import pickle
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def parse_sweep_ids(value):
    """Parse a single sweep ID or a comma-separated string of sweep IDs."""
    sweep_ids = [sweep_id.strip() for sweep_id in value.split(",")]
    if not all(sweep_ids):
        raise ValueError("Sweep IDs must be nonempty (use comma-separated IDs).")
    return sweep_ids


def collect_merged_sweep_data(api, entity, project, sweep_ids, cache_dir, method_key,
                              no_cache, sweep_vars, sweep_labels, fast=False,
                              download_workers=4, history_page_size=10000):
    """Combine sweeps, reporting overlapping grid points and avoiding repeated runs."""
    results = []
    seen_sweeps = set()
    seen_runs = {}
    for sweep_id in sweep_ids:
        if sweep_id in seen_sweeps:
            warnings.warn(f"Duplicate sweep ID {sweep_id} for {METHOD_LABELS[method_key]}; skipping.",
                          stacklevel=2)
            continue
        seen_sweeps.add(sweep_id)
        runs = collect_sweep_data(api, entity, project, sweep_id, cache_dir, method_key,
                                  no_cache, sweep_vars, sweep_labels, fast=fast,
                                  download_workers=download_workers, history_page_size=history_page_size)
        # Compare across sweeps; a grid point overlaps only for the same seed.
        previous_runs = list(results)
        for run in runs:
            run_id = run["run_id"]
            if run_id in seen_runs:
                warnings.warn(
                    f"Duplicate run {run_id} in sweeps {seen_runs[run_id]} and {sweep_id} "
                    f"for {METHOD_LABELS[method_key]}; skipping repeated run.", stacklevel=2)
                continue
            value = run_sweep_value(run, sweep_vars)
            overlaps = [other for other in previous_runs
                        if run_sweep_value(other, sweep_vars) == value
                        and other.get("seed") == run.get("seed")]
            if overlaps:
                sources = ", ".join(
                    f"{seen_runs[other['run_id']]}/{other['run_id']} (seed={other.get('seed')})"
                    for other in overlaps
                )
                warnings.warn(
                    f"Duplicate sweep grid point for {METHOD_LABELS[method_key]}: "
                    f"{format_sweep_value(value, sweep_labels)}; "
                    f"{sweep_id}/{run_id} (seed={run.get('seed')}) overlaps {sources}. "
                    "Keeping distinct runs in the merged data.", stacklevel=2)
            seen_runs[run_id] = sweep_id
            results.append(run)
    return results


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
    """Sort scalar sweep values, retaining order for non-scalar values."""
    values = list(values)
    if any(isinstance(value, (list, dict, set)) for value in values):
        return values
    try:
        return sorted(values, key=float)
    except (TypeError, ValueError):
        return sorted(values, key=lambda value: str(value))


def unique_sweep_values(values):
    """Return values in first-seen order, including unhashable config values."""
    unique_values = []
    for value in values:
        if value not in unique_values:
            unique_values.append(value)
    return unique_values


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


TRAIN_KEYS = ["train/integration_step", "train/batch_final_accuracy", "train/batch_final_error", "epoch", "train/sample"]


def _is_nan(v):
    if v is None:
        return False
    try:
        return np.isnan(float(v))
    except (ValueError, TypeError):
        return False


def latest_test_history(run, history_page_size=10000):
    """Fetch the last logged test accuracy, searching only the history tail first."""
    end = run.lastHistoryStep + 1
    window = 1000
    while end > 0:
        start = max(0, end - window)
        rows = list(run.scan_history(
            keys=["test/avg_epoch_accuracy"], min_step=start, max_step=end,
            page_size=history_page_size,
        ))
        if rows:
            return rows[-1:]
        end = start
        window *= 2
    return []


def download_run_data(run, cache_dir, no_cache, sweep_vars, pbar=None, fast=False,
                      history_page_size=10000):
    cache_path = cache_dir / f"{run.id}.pkl"
    # Summary-only downloads must never masquerade as full-history caches.
    if fast and not (not no_cache and cache_path.exists()):
        cache_path = cache_dir / f"{run.id}.fast.pkl"
    if not no_cache and cache_path.exists():
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
        if (fast or set(TRAIN_KEYS).issubset(data.get("train_history_keys", []))
                or all("train/sample" in row for row in data["train_history"])):
            if pbar is not None:
                pbar.set_postfix_str(f"cached {run.id[:8]}")
            return data
        # Older caches did not request the cumulative sample index.
        # Reload the history so sample-count plots also work with existing caches.
        print(f"\n  Refreshing cache for {run.id}: old history lacks train/sample")

    if pbar is not None:
        pbar.set_postfix_str(f"downloading {run.id[:8]}")
    train_history = []
    for row in (() if fast else run.scan_history(keys=TRAIN_KEYS, page_size=history_page_size)):
        if "train/integration_step" in row and "train/batch_final_accuracy" in row:
            train_history.append(row)
        if pbar is not None:
            pbar.update(1)

    test_acc = run.summary.get("test/avg_epoch_accuracy")
    test_epoch = run.summary.get("epoch")
    if test_acc is None:
        # Search backwards in bounded windows if the summary is missing.
        # Test metrics can be logged before the run's very last history row.
        for row in (latest_test_history(run, history_page_size) if fast else
                    run.scan_history(keys=["test/avg_epoch_accuracy", "epoch"],
                                     page_size=history_page_size)):
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
        # Record the request even if this run never logged every requested field.
        "train_history_keys": [] if fast else list(TRAIN_KEYS),
        "has_nan_loss": _is_nan(run.summary.get("train/batch_final_error")) if fast else any(
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


def collect_sweep_data(api, entity, project, sweep_id, cache_dir, method_key, no_cache, sweep_vars, sweep_labels, fast=False,
                              download_workers=4, history_page_size=10000):
    print(f"\n{METHOD_LABELS[method_key]} (sweep {sweep_id})")
    # Sweep.runs paginates without an explicit order. Page boundaries can then
    # repeat runs and omit others, biasing particular sweep values and seeds.
    runs = list(api.runs(
        f"{entity}/{project}", filters={"sweep": sweep_id},
        order="+created_at", per_page=100,
    ))
    seen_ids = set()
    for run in runs:
        if run.id in seen_ids:
            raise ValueError(
                f"Sweep {sweep_id} returned duplicate run {run.id}; "
                "the run listing may also be missing runs. Retry the query."
            )
        seen_ids.add(run.id)
    print(f"  {len(runs)} runs in sweep")
    if download_workers < 1 or history_page_size < 1:
        raise ValueError("download_workers and history_page_size must be positive integers.")

    def download(run):
        return download_run_data(
            run, cache_dir, no_cache, sweep_vars, fast=fast,
            history_page_size=history_page_size,
        )

    # Each run is handled by exactly one worker. Keep results in listing order,
    # independent of completion order, so seed and sweep merging stay stable.
    downloaded = [None] * len(runs)
    with ThreadPoolExecutor(max_workers=download_workers) as executor:
        futures = {executor.submit(download, run): idx for idx, run in enumerate(runs)}
        with tqdm(total=len(runs), desc="Downloading/caching runs", unit="run") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    downloaded[idx] = future.result()
                except Exception as exc:
                    for pending in futures:
                        pending.cancel()
                    raise RuntimeError(f"Failed to download run {runs[idx].id}") from exc
                pbar.set_postfix_str(runs[idx].id[:8])
                pbar.update(1)

    results = []
    for run, data in zip(runs, downloaded):
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


def _accuracy_vs_integration_step_common(all_data, sweep_vars, sweep_values, x_axis="Integration Step"):
    n_values = len(sweep_values)
    if n_values == 0:
        print(f"WARNING: No values found for {sweep_vars}, skipping accuracy vs {x_axis.lower()} plot.")
        return None

    n_cols = min(4, n_values)
    n_rows = (n_values + n_cols - 1) // n_cols

    method_order = [METHOD_LABELS[key] for key in METHOD_KEYS if key in all_data]
    palette = METHOD_PALETTE

    all_records = []
    nan_methods_by_index = {}
    for sweep_idx, sweep_value in enumerate(sweep_values):
        nan_methods_by_index[sweep_idx] = []
        for method_key in METHOD_KEYS:
            if method_key not in all_data:
                continue
            runs = find_runs_for_sweep_value(all_data.get(method_key, []), sweep_vars, sweep_value, exclude_nan=False)
            nan_runs = [r for r in runs if r["has_nan_loss"]]
            valid_runs = [r for r in runs if not r["has_nan_loss"]]
            if nan_runs and not valid_runs:
                nan_methods_by_index[sweep_idx].append(METHOD_LABELS[method_key])
            for r in valid_runs:
                # Training epochs are one-based, so the logged index includes
                # an extra epoch. Align each run's first recorded sample to 1.
                sample_origin = min(
                    (h["train/sample"] for h in r["train_history"]
                     if h.get("train/sample") is not None), default=0,
                ) if x_axis == "Sample Count" else 0
                for h in r["train_history"]:
                    if x_axis == "Sample Count":
                        if h.get("train/sample") is None:
                            continue
                        x_value = h["train/sample"] - sample_origin + 1
                    else:
                        x_value = h["train/integration_step"]
                    all_records.append({
                        "Sweep Index": sweep_idx,
                        x_axis: x_value,
                        "Accuracy (%)": h["train/batch_final_accuracy"] * 100,
                        "Method": METHOD_LABELS[method_key],
                        "seed": r.get("seed") if r.get("seed") is not None else r["run_id"],
                    })

    df_all = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    return n_values, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_index


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


def plot_accuracy_vs_integration_step_mean(all_data, sweep_vars, sweep_labels, sweep_values, output_dir, dpi=200,
                                                  x_axis="Integration Step"):
    result = _accuracy_vs_integration_step_common(all_data, sweep_vars, sweep_values, x_axis)
    if result is None:
        return
    n_values, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_index = result

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, sweep_value in enumerate(sweep_values):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_value = df_all[df_all["Sweep Index"] == idx]
            if len(df_value) > 0:
                n_seeds = df_value.groupby(["Method", x_axis]).size().groupby("Method").first()
                has_multi = (n_seeds > 1).any()
                sns.lineplot(
                    data=df_value, x=x_axis, y="Accuracy (%)",
                    hue="Method", hue_order=method_order, palette=palette,
                    ax=ax, legend=False,
                    errorbar="sd" if has_multi else None,
                )
                if x_axis == "Sample Count":
                    ax.set_xlim(0, max(1, df_value[x_axis].max()) * 1.02)
                else:
                    mean_curves = []
                    for m in method_order:
                        df_m = df_value[df_value["Method"] == m]
                        if len(df_m) > 0:
                            means = df_m.groupby(x_axis)["Accuracy (%)"].mean()
                            mean_curves.append((means.index.values, means.values))
                    if mean_curves:
                        ax.set_xlim(0, _early_segment_xlim(mean_curves))

        for m in nan_methods_by_index[idx]:
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(format_sweep_value(sweep_value, sweep_labels), fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_index[idx]:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    _finalize_accuracy_subplot(fig, axes, n_values, n_cols, n_rows, method_order, palette,
                               f"Training Accuracy vs. {x_axis} (mean ± SD)",
                               output_dir / f"accuracy_vs_{x_axis.lower().replace(' ', '_')}_mean", dpi)


def plot_accuracy_vs_integration_step_seeds(all_data, sweep_vars, sweep_labels, sweep_values, output_dir, dpi=200,
                                                  x_axis="Integration Step"):
    result = _accuracy_vs_integration_step_common(all_data, sweep_vars, sweep_values, x_axis)
    if result is None:
        return
    n_values, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_index = result

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, sweep_value in enumerate(sweep_values):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_value = df_all[df_all["Sweep Index"] == idx]
            if len(df_value) > 0:
                sns.lineplot(
                    data=df_value, x=x_axis, y="Accuracy (%)",
                    hue="Method", hue_order=method_order, palette=palette,
                    ax=ax, legend=False,
                    units="seed", estimator=None, alpha=0.4,
                )
                if x_axis == "Sample Count":
                    ax.set_xlim(0, max(1, df_value[x_axis].max()) * 1.02)
                else:
                    mean_curves = []
                    for m in method_order:
                        df_m = df_value[df_value["Method"] == m]
                        if len(df_m) > 0:
                            means = df_m.groupby(x_axis)["Accuracy (%)"].mean()
                            mean_curves.append((means.index.values, means.values))
                    if mean_curves:
                        ax.set_xlim(0, _early_segment_xlim(mean_curves))

        for m in nan_methods_by_index[idx]:
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(format_sweep_value(sweep_value, sweep_labels), fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_index[idx]:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    _finalize_accuracy_subplot(fig, axes, n_values, n_cols, n_rows, method_order, palette,
                               f"Training Accuracy vs. {x_axis} (individual seeds)",
                               output_dir / f"accuracy_vs_{x_axis.lower().replace(' ', '_')}_seeds", dpi)


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


def plot_final_test_accuracy_bar(all_data, sweep_vars, sweep_labels, sweep_values, output_dir, dpi=200,
                                 baselines=None):
    if not sweep_values:
        print(f"WARNING: No values found for {sweep_vars}, skipping final test accuracy plot.")
        return

    x_values = sort_sweep_values(unique_sweep_values(value[0] for value in sweep_values))
    facet_values = (sort_sweep_values(unique_sweep_values(value[1] for value in sweep_values))
                    if len(sweep_vars) == 2 else [None])
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
        for baseline_idx, (label, mean) in enumerate((baselines or {}).items()):
            ax.axhline(mean, label=label, color=sns.color_palette("colorblind", 6)[3 + baseline_idx],
                       linestyle="--", linewidth=1.5)
        if baselines and legend_ax is None:
            legend_ax = ax
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
    for ax in axes.flat:
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    fig.tight_layout()
    if handles:
        baseline_labels = set(baselines or {})
        method_entries = [(handle, label) for handle, label in zip(handles, labels)
                          if label not in baseline_labels]
        baseline_entries = [(handle, label) for handle, label in zip(handles, labels)
                            if label in baseline_labels]
        legend_y = -0.02
        for entries in (method_entries, baseline_entries):
            if entries:
                row_handles, row_labels = zip(*entries)
                legend = fig.legend(row_handles, row_labels, frameon=False,
                                    loc="upper center", ncol=len(entries),
                                    bbox_to_anchor=(0.5, legend_y))
                fig.canvas.draw()
                legend_y -= legend.get_window_extent().height / fig.bbox.height + 0.01
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
    baselines = {
        label: float(cfg[key])
        for key, label in [("bp", "Backpropagation"), ("dfc", "DFC"), ("strong-dfc", "Strong-DFC")]
        if cfg.get(key) is not None
    }

    for key in ("download_workers", "history_page_size"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")

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
        all_data[method_key] = collect_merged_sweep_data(
            api, cfg.entity, cfg.project, parse_sweep_ids(sweep_id), cache_dir, method_key, cfg.no_cache,
            sweep_vars, sweep_labels, fast=cfg.fast,
            download_workers=cfg.download_workers, history_page_size=cfg.history_page_size
        )

    values_by_var = [sort_sweep_values(unique_sweep_values(
                        run_data["sweep_values"][sweep_var]
                        for runs in all_data.values() for run_data in runs
                    ))
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

    if not cfg.fast:
        plot_accuracy_vs_integration_step_mean(
            all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi
        )
        plot_accuracy_vs_integration_step_seeds(
            all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi
        )
        plot_accuracy_vs_integration_step_mean(
            all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi,
            x_axis="Sample Count",
        )
        plot_accuracy_vs_integration_step_seeds(
            all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi,
            x_axis="Sample Count",
        )
    plot_final_test_accuracy_bar(
        all_data, sweep_vars, sweep_labels, sweep_values, output_dir, cfg.dpi, baselines=baselines
    )
    print_final_test_accuracy_summary(all_data, sweep_vars, sweep_labels, sweep_values)


if __name__ == "__main__":
    main()
