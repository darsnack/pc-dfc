"""Relay one manifest-backed sweep continuation run to its original program."""

import json
import os
import subprocess
import sys
from pathlib import Path


RELAUNCH_KEYS = ("relaunch_manifest", "relaunch_source_run_id")


def parse_assignments(argv: list[str]) -> dict[str, str]:
    values = {}
    for argument in argv:
        key, separator, value = argument.partition("=")
        if separator and key in RELAUNCH_KEYS:
            values[key] = value
    missing = [key for key in RELAUNCH_KEYS if key not in values]
    if missing:
        raise ValueError(f"missing continuation argument(s): {', '.join(missing)}")
    return values


def flatten_config(value, prefix: str = ""):
    if isinstance(value, dict):
        for key, nested_value in value.items():
            child = f"{prefix}.{key}" if prefix else key
            yield from flatten_config(nested_value, child)
    elif prefix:
        yield prefix, value


def hydra_value(value) -> str:
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def relay_command(entry: dict) -> tuple[list[str], dict[str, str]]:
    command_spec = entry.get("command")
    program = entry["program"]
    env = os.environ.copy()
    command = []

    # Source sweep commands use W&B macros. Preserve literal environment
    # assignments and executable tokens; replace its program/args macros below.
    if isinstance(command_spec, list):
        for token in command_spec:
            if token == "${env}" or token.startswith("${args"):
                continue
            if token == "${program}":
                command.append(program)
            elif isinstance(token, str) and "=" in token and not command:
                key, value = token.split("=", 1)
                if key.isidentifier():
                    env[key] = value
                else:
                    command.append(token)
            else:
                command.append(token)
    if not command:
        command = ["uv", "run", program]
    elif program not in command:
        command.append(program)

    overrides = [
        f"{key}={hydra_value(value)}"
        for key, value in flatten_config(entry["config"])
        if key != "checkpointing.load_from"
    ]
    overrides.append(f"checkpointing.load_from={hydra_value(entry['checkpoint_path'])}")
    return [*command, *overrides], env


def main(argv: list[str]) -> int:
    args = parse_assignments(argv)
    manifest_path = Path(args["relaunch_manifest"])
    manifest = json.loads(manifest_path.read_text())
    entries = {entry["source_run_id"]: entry for entry in manifest["entries"]}
    try:
        entry = entries[args["relaunch_source_run_id"]]
    except KeyError as exc:
        raise ValueError(f"source run {exc.args[0]!r} is not in {manifest_path}") from exc
    command, env = relay_command(entry)
    print(f"Continuing source run {entry['source_run_id']} from {entry['checkpoint_path']}", flush=True)
    return subprocess.run(command, env=env).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
