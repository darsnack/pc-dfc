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

    data = {
        "lr": lr,
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
            print(f"  {run.id[:8]}  LR={data['lr']:.1e}  train_pts={n_train}  {test_str}")
            results.append(data)
    print(f"  Total: {len(results)}/{len(runs)} runs for {METHOD_LABELS[method_key]}")
    return results


def lr_match(a, b, tol=1e-6):
    if a == 0 and b == 0:
        return True
    if a == 0 or b == 0:
        return False
    return abs(a - b) / max(abs(a), abs(b)) < tol


def find_run_for_lr(runs, lr):
    for run_data in runs:
        if lr_match(run_data["lr"], lr):
            return run_data
    return None


def plot_accuracy_vs_integration_step(all_data, lrs, output_dir):
    n_lrs = len(lrs)
    if n_lrs == 0:
        print("WARNING: No learning rates found, skipping accuracy vs integration step plot.")
        return
    n_cols = min(4, n_lrs)
    n_rows = (n_lrs + n_cols - 1) // n_cols

    method_order = [METHOD_LABELS[k] for k in ("pc", "approx_pc", "pc_nudges")]
    palette = dict(zip(method_order, sns.color_palette(n_colors=len(method_order))))

    all_records = []
    nan_methods_by_lr = {}
    for lr in lrs:
        nan_methods_by_lr[lr] = []
        for method_key in ("pc", "approx_pc", "pc_nudges"):
            run_data = find_run_for_lr(all_data.get(method_key, []), lr)
            if run_data is None:
                continue
            if run_data["has_nan_loss"]:
                nan_methods_by_lr[lr].append(METHOD_LABELS[method_key])
                continue
            for r in run_data["train_history"]:
                all_records.append({
                    "lr": lr,
                    "Integration Step": r["train/integration_step"],
                    "Accuracy (%)": r["train/batch_final_accuracy"] * 100,
                    "Method": METHOD_LABELS[method_key],
                })

    df_all = pd.DataFrame(all_records) if all_records else pd.DataFrame()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, lr in enumerate(lrs):
        ax = axes[idx // n_cols][idx % n_cols]

        if not df_all.empty:
            df_lr = df_all[df_all["lr"] == lr]
            if len(df_lr) > 0:
                sns.lineplot(data=df_lr, x="Integration Step", y="Accuracy (%)",
                             hue="Method", hue_order=method_order, palette=palette,
                             ax=ax, legend=False)
                curves = []
                for method_key in ("pc", "approx_pc", "pc_nudges"):
                    run_data = find_run_for_lr(all_data.get(method_key, []), lr)
                    if run_data is not None and not run_data["has_nan_loss"]:
                        steps = [r["train/integration_step"] for r in run_data["train_history"]]
                        accs = [r["train/batch_final_accuracy"] * 100 for r in run_data["train_history"]]
                        curves.append((np.array(steps, dtype=float), np.array(accs, dtype=float)))
                if curves:
                    ax.set_xlim(0, _early_segment_xlim(curves))

        for m in nan_methods_by_lr.get(lr, []):
            ax.plot([], [], color=palette[m], label=f"{m} (diverged)")

        ax.set_title(f"LR = {lr:.1e}", fontweight="bold")
        ax.set_ylim(-2, 102)

        if df_all.empty and not nan_methods_by_lr.get(lr):
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        sns.despine(ax=ax)

    for idx in range(n_lrs, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    handles, labels = [], []
    for m in method_order:
        handles.append(plt.Line2D([0], [0], color=palette[m]))
        labels.append(m)
    fig.legend(handles, labels, frameon=False, loc="upper center",
               ncol=len(method_order), bbox_to_anchor=(0.5, 1.0))

    fig.suptitle("Training Accuracy vs. Integration Step (early segment)",
                 fontweight="bold", y=1.04)
    fig.tight_layout()

    out_path = output_dir / "accuracy_vs_integration_step"
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=200)
    print(f"Saved {out_path}.png")
    plt.close(fig)


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


def plot_final_test_accuracy_bar(all_data, lrs, output_dir):
    n_lrs = len(lrs)
    if n_lrs == 0:
        print("WARNING: No learning rates found, skipping final test accuracy bar plot.")
        return

    records = []
    nan_records = []
    for lr in lrs:
        for method_key in ("pc", "approx_pc", "pc_nudges"):
            run_data = find_run_for_lr(all_data.get(method_key, []), lr)
            if run_data is None:
                continue
            if run_data["has_nan_loss"]:
                nan_records.append({
                    "Learning Rate": f"{lr:.1e}",
                    "Method": METHOD_LABELS[method_key],
                })
            elif run_data["test_accuracy"] is not None:
                records.append({
                    "Learning Rate": f"{lr:.1e}",
                    "Method": METHOD_LABELS[method_key],
                    "Accuracy (%)": run_data["test_accuracy"] * 100,
                })

    if not records and not nan_records:
        print("WARNING: No data for bar plot.")
        return

    df = pd.DataFrame(records) if records else pd.DataFrame(columns=["Learning Rate", "Method", "Accuracy (%)"])
    df_nan = pd.DataFrame(nan_records) if nan_records else pd.DataFrame(columns=["Learning Rate", "Method"])

    lr_order = [f"{lr:.1e}" for lr in lrs]
    method_order = [METHOD_LABELS[k] for k in ("pc", "approx_pc", "pc_nudges")]

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * n_lrs), 5))
    sns.barplot(data=df, x="Learning Rate", y="Accuracy (%)", hue="Method",
                order=lr_order, hue_order=method_order, ax=ax, saturation=1, edgecolor="none")

    for lr_str in lr_order:
        for _, row in df_nan[df_nan["Learning Rate"] == lr_str].iterrows():
            cat_idx = lr_order.index(lr_str)
            method_idx = method_order.index(row["Method"])
            n_methods = sum(1 for m in method_order
                           if any(r["Method"] == m and r["Learning Rate"] == lr_str
                                  for r in nan_records) or
                           any(r["Method"] == m and r["Learning Rate"] == lr_str
                                  for r in records))
            offset = (method_idx - (n_methods - 1) / 2) * (0.8 / len(method_order))
            ax.text(cat_idx + offset, 2, r"$\times$", ha="center", va="bottom",
                    color="black")

    ax.set_title("Final Test Accuracy by Method and Learning Rate", fontweight="bold")
    ax.set_ylim(0, 105)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), frameon=False)
    sns.despine(ax=ax)

    fig.tight_layout()

    out_path = output_dir / "final_test_accuracy_bar"
    fig.savefig(f"{out_path}.png", bbox_inches="tight", dpi=200)
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

    plot_accuracy_vs_integration_step(all_data, lrs, output_dir)
    plot_final_test_accuracy_bar(all_data, lrs, output_dir)


if __name__ == "__main__":
    main()
