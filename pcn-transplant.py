"""Run evaluation-only PCN transplant experiments for every run in a W&B sweep.

Usage:
    uv run pcn-transplant.py --sweep-id <source-sweep-id> \
        [--entity <entity>] [--project <project>] [--dry-run]
"""

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from glob import glob
from pathlib import Path

import wandb


# Orbax writes this when its CheckpointManager creates the directory. It is
# present in the W&B output.log for each training run.
CHECKPOINT_DIR_RE = re.compile(r"Created directory=(?P<path>.+?/checkpoints)(?:[\s.:]|$)")
SOURCE_CONFIG_KEYS = ("data", "dynamics", "loss", "model", "nmodels", "nudge", "training")


def get_config_value(config, key):
    """Read a W&B config value stored either flat or as nested dictionaries."""
    if key in config:
        return config[key]

    value = config
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def get_run_log(run, download_dir):
    """Return a run's output log, preferring the local W&B log when available."""
    local_logs = sorted(glob(f"wandb/run-*-{run.id}/files/output.log"))
    if local_logs:
        try:
            return Path(local_logs[-1]).read_text(errors="replace")
        except OSError as exc:
            raise RuntimeError(f"could not read local output.log: {exc}") from exc

    try:
        log_file = run.file("output.log")
        downloaded = log_file.download(root=download_dir, replace=True)
    except Exception as exc:
        raise RuntimeError(f"could not download output.log: {exc}") from exc

    try:
        return Path(downloaded.name).read_text(errors="replace")
    except (AttributeError, OSError) as exc:
        raise RuntimeError(f"could not read downloaded output.log: {exc}") from exc


def checkpoint_dirs_from_log(log_text):
    """Return checkpoint directories mentioned by Orbax, preserving log order."""
    return list(dict.fromkeys(match.group("path") for match in CHECKPOINT_DIR_RE.finditer(log_text)))


def format_override_value(value):
    """Format a W&B config value for a Hydra command-line override."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ","))
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def quote_hydra_string(value):
    """Quote a literal string so Hydra does not parse path characters as syntax."""
    return json.dumps(str(value))


def source_config_overrides(config):
    """Return the model and evaluation settings needed to reproduce a source run."""
    overrides = []
    for key in SOURCE_CONFIG_KEYS:
        value = get_config_value(config, key)
        if value is None:
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in sorted(value.items()):
                overrides.append(f"{key}.{nested_key}={format_override_value(nested_value)}")
        else:
            overrides.append(f"{key}={format_override_value(value)}")
    return overrides


def collect_source_runs(api, entity, project, sweep_id):
    """Resolve each sweep run to its locally accessible checkpoint directory."""
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    runs = list(sweep.runs)
    print(f"Source sweep {sweep_id}: {len(runs)} runs")

    sources = []
    with tempfile.TemporaryDirectory(prefix="pcn-transplant-wandb-") as download_dir:
        for index, run in enumerate(runs, start=1):
            try:
                log_text = get_run_log(run, download_dir)
                checkpoint_dirs = checkpoint_dirs_from_log(log_text)
            except RuntimeError as exc:
                print(f"WARNING: skipping {run.id}: {exc}", file=sys.stderr)
                continue

            if not checkpoint_dirs:
                print(f"WARNING: skipping {run.id}: no checkpoint path in output.log", file=sys.stderr)
                continue

            checkpoint_dir = checkpoint_dirs[-1]
            if not os.path.isdir(checkpoint_dir):
                print(
                    f"WARNING: skipping {run.id}: checkpoint directory is not accessible: {checkpoint_dir}",
                    file=sys.stderr,
                )
                continue

            learning_rate = get_config_value(run.config, "training.learning_rate")
            seed = get_config_value(run.config, "seed")
            if learning_rate is None or seed is None:
                print(
                    f"WARNING: skipping {run.id}: missing training.learning_rate or seed in W&B config",
                    file=sys.stderr,
                )
                continue

            try:
                seed = int(seed)
            except (TypeError, ValueError):
                print(f"WARNING: skipping {run.id}: invalid seed {seed!r}", file=sys.stderr)
                continue

            print(f"  {index}/{len(runs)} {run.id}: {checkpoint_dir}")
            sources.append({
                "run_id": run.id,
                "checkpoint_dir": checkpoint_dir,
                "seed": seed,
                "config_overrides": source_config_overrides(run.config),
            })

    if not sources:
        raise RuntimeError(f"no accessible checkpoints found in sweep {sweep_id}")
    return sources


def transplant_command(source, sweep_id):
    """Build the existing transplant command, replacing training with evaluation."""
    return [
        "uv", "run", "train-pcn.py",
        "checkpointing.load_from="
        f"{quote_hydra_string(os.path.abspath(source['checkpoint_dir']))}",
        *source["config_overrides"],
        "training.epochs=0",
        f"seed={source['seed']}",
        "hydra.run.dir="
        f"./outputs/pcn_transplant/sweep_{sweep_id}/run_{source['run_id']}"
        "/${now:%Y-%m-%d}/${now:%H-%M-%S}",
    ]


def main():
    parser = argparse.ArgumentParser(description="PCN transplant experiment for a W&B source sweep")
    parser.add_argument("--sweep-id", required=True, help="W&B source sweep ID")
    parser.add_argument("--entity", default="mitra-ryan-kyle", help="W&B entity")
    parser.add_argument("--project", default="pc-dfc", help="W&B project")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    args = parser.parse_args()

    api = wandb.Api(timeout=60)
    try:
        sources = collect_source_runs(api, args.entity, args.project, args.sweep_id)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    for source in sources:
        cmd = transplant_command(source, args.sweep_id)
        print(f"Command: {' '.join(shlex.quote(arg) for arg in cmd)}")
        if args.dry_run:
            continue

        result = subprocess.run(cmd)
        if result.returncode:
            sys.exit(result.returncode)


if __name__ == "__main__":
    main()
