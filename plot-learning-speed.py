"""Plot learning speed comparison across PC, Approx PC, and PC with Nudges.

Downloads per-sample training data and test accuracy from wandb sweeps,
caches it locally, and produces:
  1. Training batch final accuracy vs. integration step (one subplot per LR)
  2. Grouped bar chart of final test accuracy across methods and LRs

Usage:
  uv run plot-learning-speed.py pc_sweep_id=??? approx_pc_sweep_id=??? pc_nudges_sweep_id=??? \
      [entity=...] [project=...] [cache_dir=...] [no_cache=...]
"""

import pickle
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


def get_lr_from_config(config):
    if "training" in config and isinstance(config["training"], dict):
        v = config["training"].get("learning_rate")
        if v is not None:
            return float(v)
    if "training.learning_rate" in config:
        return float(config["training.learning_rate"])
    for k, v in config.items():
        if "learning_rate" in str(k).lower():
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


TRAIN_KEYS = ["train/integration_step", "train/batch_final_accuracy", "train/batch_final_error", "epoch"]


def _is_nan(v):
    if v is None:
        return False
    try:
        return np.isnan(float(v))
    except (ValueError, TypeError):
        return False


def download_run_data(run, cache_dir, no_cache, pbar=None):
    cache_path = cache_dir / f"{run.id}.pkl"
    if not no_cache and cache_path.exists():
        if pbar is not None:
            pbar.set_postfix_str(f"cached {run.id[:8]}")
        with cache_path.open("rb") as f:
            return pickle.load(f)

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

    lr = get_lr_from_config(run.config)
    if lr is None:
        print(f"\n  WARNING: Could not find learning rate for run {run.id}")
        print(f"    Config keys: {list(run.config.keys())}")
        return None

    seed = run.config.get("seed")
    if seed is not None:
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            seed = None

    data = {
        "lr": lr,
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


def collect_sweep_data(api, entity, project, sweep_id, cache_dir, method_key, no_cache):
    print(f"\n{METHOD_LABELS[method_key]} (sweep {sweep_id})")
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = list(sweep.runs)
    print(f"  {len(runs)} runs in sweep")
    results = []
    for run_idx, run in enumerate(runs):
        lr = get_lr_from_config(run.config)
        lr_str = f"{lr:.1e}" if lr is not None else "???"
        label = f"run {run_idx + 1}/{len(runs)} LR={lr_str}"
        pbar = tqdm(desc=label, unit=" rows", leave=True)
        data = download_run_data(run, cache_dir, no_cache, pbar)
        pbar.close()
        if data is not None:
            data["method"] = method_key
            n_train = len(data["train_history"])
            test_str = f"test_acc={data['test_accuracy']:.3f}" if data["test_accuracy"] is not None else "no test acc"
            seed_str = f"seed={data['seed']}" if data.get("seed") is not None else "no seed"
            print(f"  {run.id[:8]}  LR={data['lr']:.1e}  {seed_str}  train_pts={n_train}  {test_str}")
            results.append(data)
    print(f"  Total: {len(results)}/{len(runs)} runs for {METHOD_LABELS[method_key]}")
    return results


def lr_match(a, b, tol=1e-6):
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) < tol


def find_runs_for_lr(runs, lr, exclude_nan=True):
    matched = []
    for run_data in runs:
        if lr_match(run_data["lr"], lr):
            if exclude_nan and run_data["has_nan_loss"]:
                continue
            matched.append(run_data)
    return matched


def _accuracy_vs_integration_step_common(all_data, lrs):
    n_lrs = len(lrs)
    if n_lrs == 0:
        print("WARNING: No learning rates found, skipping accuracy vs integration step plot.")
        return None

    n_cols = min(4, n_lrs)
    n_rows = (n_lrs + n_cols - 1) // n_cols

    method_order = [METHOD_LABELS[k] for k in ("pc", "approx_pc", "pc_nudges")]
    palette = dict(zip(method_order, sns.color_palette(n_colors=len(method_order))))

    all_records = []
    nan_methods_by_lr = {}
    for lr in lrs:
        nan_methods_by_lr[lr] = []
        for method_key in ("pc", "approx_pc", "pc_nudges"):
            runs = find_runs_for_lr(all_data.get(method_key, []), lr, exclude_nan=False)
            nan_runs = [r for r in runs if r["has_nan_loss"]]
            valid_runs = [r for r in runs if not r["has_nan_loss"]]
            if nan_runs and not valid_runs:
                nan_methods_by_lr[lr].append(METHOD_LABELS[method_key])
            for r in valid_runs:
                for h in r["train_history"]:
                    all_records.append({
                        "lr": lr,
                        "Integration Step": h["train/integration_step"],
                        "Accuracy (%)": h["train/batch_final_accuracy"] * 100,
                        "Method": METHOD_LABELS[method_key],
                        "seed": r.get("seed") if r.get("seed") is not None else r["run_id"],
                    })

    df_all = pd.DataFrame(all_records) if all_records else pd.DataFrame()
    return n_lrs, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_lr


def _finalize_accuracy_subplot(fig, axes, n_lrs, n_cols, n_rows, method_order, palette, title, out_path, dpi):
    for idx in range(n_lrs, n_rows * n_cols):
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


def plot_accuracy_vs_integration_step_mean(all_data, lrs, output_dir, dpi = 200):
    result = _accuracy_vs_integration_step_common(all_data, lrs)
    if result is None:
        return
    n_lrs, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_lr = result

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, lr in enumerate(lrs):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_lr = df_all[df_all["lr"] == lr]
            if len(df_lr) > 0:
                n_seeds = df_lr.groupby(["Method", "Integration Step"]).size().groupby("Method").first()
                has_multi = (n_seeds > 1).any()
                sns.lineplot(
                    data=df_lr, x="Integration Step", y="Accuracy (%)",
                    hue="Method", hue_order=method_order, palette=palette,
                    ax=ax, legend=False,
                    errorbar="sd" if has_multi else None,
                )
                mean_curves = []
                if not df_lr.empty:
                    for m in method_order:
                        df_m = df_lr[df_lr["Method"] == m]
                        if len(df_m) > 0:
                            grp = df_m.groupby("Integration Step")["Accuracy (%)"]
                            mean_accs = grp.mean().values
                            steps = grp.mean().index.values
                            mean_curves.append((steps, mean_accs))
                if mean_curves:
                    ax.set_xlim(0, _early_segment_xlim(mean_curves))

        for m in nan_methods_by_lr.get(lr, []):
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(f"LR = {lr:.1e}", fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_lr.get(lr):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    _finalize_accuracy_subplot(fig, axes, n_lrs, n_cols, n_rows, method_order, palette,
                               "Training Accuracy vs. Integration Step (mean ± SD)",
                               output_dir / "accuracy_vs_integration_step_mean", dpi)


def plot_accuracy_vs_integration_step_seeds(all_data, lrs, output_dir, dpi = 200):
    result = _accuracy_vs_integration_step_common(all_data, lrs)
    if result is None:
        return
    n_lrs, n_cols, n_rows, method_order, palette, df_all, nan_methods_by_lr = result

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    for idx, lr in enumerate(lrs):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_lr = df_all[df_all["lr"] == lr]
            if len(df_lr) > 0:
                sns.lineplot(
                    data=df_lr, x="Integration Step", y="Accuracy (%)",
                    hue="Method", hue_order=method_order, palette=palette,
                    units="seed", estimator=None,
                    ax=ax, legend=False, alpha=0.4,
                )
                mean_curves = []
                if not df_lr.empty:
                    for m in method_order:
                        df_m = df_lr[df_lr["Method"] == m]
                        if len(df_m) > 0:
                            grp = df_m.groupby("Integration Step")["Accuracy (%)"]
                            mean_accs = grp.mean().values
                            steps = grp.mean().index.values
                            mean_curves.append((steps, mean_accs))
                if mean_curves:
                    ax.set_xlim(0, _early_segment_xlim(mean_curves))

        for m in nan_methods_by_lr.get(lr, []):
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(f"LR = {lr:.1e}", fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_lr.get(lr):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    _finalize_accuracy_subplot(fig, axes, n_lrs, n_cols, n_rows, method_order, palette,
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


def plot_final_test_accuracy_bar(all_data, lrs, output_dir, dpi = 200):
    n_lrs = len(lrs)
    if n_lrs == 0:
        print("WARNING: No learning rates found, skipping final test accuracy bar plot.")
        return

    records = []
    nan_records = []
    for lr in lrs:
        for method_key in ("pc", "approx_pc", "pc_nudges"):
            all_runs = find_runs_for_lr(all_data.get(method_key, []), lr, exclude_nan=False)
            valid_runs = find_runs_for_lr(all_data.get(method_key, []), lr, exclude_nan=True)
            nan_runs = [r for r in all_runs if r["has_nan_loss"]]
            if nan_runs and not valid_runs:
                nan_records.append({
                    "Learning Rate": f"{lr:.1e}",
                    "Method": METHOD_LABELS[method_key],
                })
            for r in valid_runs:
                if r["test_accuracy"] is not None:
                    records.append({
                        "Learning Rate": f"{lr:.1e}",
                        "Method": METHOD_LABELS[method_key],
                        "Accuracy (%)": r["test_accuracy"] * 100,
                    })

    if not records and not nan_records:
        print("WARNING: No data for bar plot.")
        return

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["Learning Rate", "Method", "Accuracy (%)"])
    df_nan = pd.DataFrame(nan_records) if nan_records else pd.DataFrame(columns=["Learning Rate", "Method"])

    lr_order = [f"{lr:.1e}" for lr in lrs]
    method_order = [METHOD_LABELS[k] for k in ("pc", "approx_pc", "pc_nudges")]

    fig, ax = plt.subplots(figsize=(max(5.33, n_lrs), 4))
    n_seeds_per_group = df.groupby(["Learning Rate", "Method"]).size()
    has_multi = n_seeds_per_group.max() > 1 if len(n_seeds_per_group) > 0 else False
    pointplot_dodge = 0.525
    stripplot_dodge = 0.3
    sns.pointplot(data=df, x="Learning Rate", y="Accuracy (%)", hue="Method",
                  order=lr_order, hue_order=method_order, ax=ax,
                  errorbar="sd" if has_multi else None,
                  dodge=pointplot_dodge, markers="_", linestyle="none",
                  capsize=0.08, markersize=12)
    sns.stripplot(data=df, x="Learning Rate", y="Accuracy (%)", hue="Method",
                  order=lr_order, hue_order=method_order, ax=ax,
                  dodge=stripplot_dodge, alpha=0.5, size=6,
                  legend=False)

    for lr_str in lr_order:
        for _, row in df_nan[df_nan["Learning Rate"] == lr_str].iterrows():
            cat_idx = lr_order.index(lr_str)
            method_idx = method_order.index(row["Method"])
            n_methods = sum(1 for m in method_order
                           if any(r["Method"] == m and r["Learning Rate"] == lr_str
                                  for r in nan_records) or
                           any(r["Method"] == m and r["Learning Rate"] == lr_str
                                  for r in records))
            offset = (method_idx - (n_methods - 1) / 2) * 0.8 / len(method_order)
            ax.text(cat_idx + offset, 2, r"$\times$", ha="center", va="bottom",
                    color="black")

    # ax.set_title("Final Test Accuracy by Method and Learning Rate", fontweight="bold")
    ax.set_ylim(0, 105)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    sns.despine(ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()

    out_path = output_dir / "final_test_accuracy_point"
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=dpi)
    print(f"Saved {out_path}.png")
    plt.close(fig)


@hydra.main(config_path="./configs", config_name="plot-learning-speed", version_base=None)
def main(cfg: DictConfig):
    output_dir = Path.cwd()
    cache_dir = Path(get_original_cwd()) / cfg.cache_dir

    sns.set_context("paper", font_scale=2)
    sns.set_style("ticks")
    sns.set_palette("colorblind")

    api = wandb.Api(timeout=60)

    all_data = {}
    for method_key, sweep_id in [
        ("pc", cfg.pc_sweep_id),
        ("approx_pc", cfg.approx_pc_sweep_id),
        ("pc_nudges", cfg.pc_nudges_sweep_id),
    ]:
        all_data[method_key] = collect_sweep_data(
            api, cfg.entity, cfg.project, sweep_id, cache_dir, method_key, cfg.no_cache
        )

    lrs = sorted(set(run_data["lr"] for runs in all_data.values() for run_data in runs))
    print(f"\nLearning rates found: {[f'{lr:.1e}' for lr in lrs]}")

    print("\nSample sizes (total / non-NaN / with test acc):")
    for lr in lrs:
        for method_key in ("pc", "approx_pc", "pc_nudges"):
            all_runs = find_runs_for_lr(all_data.get(method_key, []), lr, exclude_nan=False)
            valid_runs = find_runs_for_lr(all_data.get(method_key, []), lr, exclude_nan=True)
            with_test = [r for r in valid_runs if r["test_accuracy"] is not None]
            print(f"  {METHOD_LABELS[method_key]:12s} LR={lr:.1e}  {len(all_runs)} / {len(valid_runs)} / {len(with_test)}")

    plot_accuracy_vs_integration_step_mean(all_data, lrs, output_dir)
    plot_accuracy_vs_integration_step_seeds(all_data, lrs, output_dir)
    plot_final_test_accuracy_bar(all_data, lrs, output_dir)


if __name__ == "__main__":
    main()
