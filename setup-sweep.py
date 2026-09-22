"""Create a W&B sweep from a YAML file or continue an existing W&B sweep."""

import argparse
import json
import os
import re
import sys
import tempfile
import uuid
from glob import glob
from pathlib import Path

import wandb
import yaml


ENTITY = "mitra-ryan-kyle"
PROJECT = "pc-dfc"
CHECKPOINT_DIR_RE = re.compile(r"Created directory=(?P<path>.+?/checkpoints)(?:[\s.:]|$)")
MANIFEST_DIR = Path("slurm-logs/continuations")


def get_run_log(run, download_dir: str) -> str:
    """Read a run's output.log, using a local W&B log when possible."""
    local_logs = sorted(glob(f"wandb/run-*-{run.id}/files/output.log"))
    if local_logs:
        return Path(local_logs[-1]).read_text(errors="replace")

    try:
        downloaded = run.file("output.log").download(root=download_dir, replace=True)
        return Path(downloaded.name).read_text(errors="replace")
    except Exception as exc:
        raise RuntimeError(f"could not retrieve output.log: {exc}") from exc


def checkpoint_dir_from_log(log_text: str) -> str | None:
    """Return the last checkpoint directory created by Orbax in a run log."""
    paths = [match.group("path") for match in CHECKPOINT_DIR_RE.finditer(log_text)]
    return paths[-1] if paths else None


def source_program(sweep_config: dict) -> str:
    program = sweep_config.get("program")
    if not isinstance(program, str) or not program:
        raise RuntimeError("source sweep has no usable 'program' field")
    return program


def continuation_manifest(api, source_id: str) -> tuple[Path, list[dict], dict]:
    """Build one continuation record for every source run with a checkpoint."""
    source_sweep = api.sweep(f"{ENTITY}/{PROJECT}/{source_id}")
    config = source_sweep.config
    program = source_program(config)
    runs = list(source_sweep.runs)
    print(f"Continuing source sweep {source_id}: {len(runs)} run(s)", file=sys.stderr)

    entries = []
    with tempfile.TemporaryDirectory(prefix="pc-dfc-sweep-resume-") as download_dir:
        for index, run in enumerate(runs, start=1):
            try:
                checkpoint_dir = checkpoint_dir_from_log(get_run_log(run, download_dir))
            except RuntimeError as exc:
                print(f"Skipping {run.id}: {exc}", file=sys.stderr)
                continue
            if not checkpoint_dir:
                print(f"Skipping {run.id}: no checkpoint path in output.log", file=sys.stderr)
                continue
            checkpoint_dir = os.path.abspath(checkpoint_dir)
            if not os.path.isdir(checkpoint_dir):
                print(f"Skipping {run.id}: checkpoint directory is not accessible: {checkpoint_dir}", file=sys.stderr)
                continue
            entries.append({
                "source_run_id": run.id,
                "program": program,
                "command": config.get("command"),
                "config": dict(run.config),
                "checkpoint_path": checkpoint_dir,
            })
            print(f"  {index}/{len(runs)} {run.id}: {checkpoint_dir}", file=sys.stderr)

    if not entries:
        raise RuntimeError(f"no accessible checkpoints found in source sweep {source_id}")

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = (MANIFEST_DIR / f"resume-{source_id}-{uuid.uuid4().hex}.json").resolve()
    manifest_path.write_text(json.dumps({
        "source_sweep_id": source_id,
        "source_sweep_config": config,
        "entries": entries,
    }, indent=2, sort_keys=True) + "\n")
    return manifest_path, entries, config


def create_continuation_sweep(source_id: str) -> tuple[str, int]:
    api = wandb.Api(timeout=60)
    manifest_path, entries, source_config = continuation_manifest(api, source_id)
    continuation_config = {
        "name": f"{source_config.get('name', source_id)}-resume-{source_id}",
        "method": "grid",
        "program": "relaunch-sweep-run.py",
        "parameters": {
            "relaunch_manifest": {"value": str(manifest_path)},
            "relaunch_source_run_id": {"values": [entry["source_run_id"] for entry in entries]},
        },
        "command": ["${env}", "uv", "run", "${program}", "${args_no_hyphens}"],
    }
    sweep_id = wandb.sweep(continuation_config, entity=ENTITY, project=PROJECT)
    return sweep_id, len(entries)


def create_sweep_from_file(sweep_file: Path) -> tuple[str, int]:
    with sweep_file.open() as file:
        sweep_config = yaml.safe_load(file)
    sweep_id = wandb.sweep(sweep_config, entity=ENTITY, project=PROJECT)
    sweep = wandb.Api(timeout=60).sweep(f"{ENTITY}/{PROJECT}/{sweep_id}")
    if not sweep.expected_run_count:
        raise RuntimeError(f"could not determine a positive run count for sweep {sweep_id}")
    return sweep_id, sweep.expected_run_count


def main(source: str) -> None:
    sweep_file = Path(source)
    if sweep_file.is_file():
        sweep_id, nruns = create_sweep_from_file(sweep_file)
    else:
        sweep_id, nruns = create_continuation_sweep(source)
    print(f"{sweep_id} {nruns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Sweep YAML file or W&B sweep ID to continue")
    args = parser.parse_args()
    try:
        main(args.source)
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
