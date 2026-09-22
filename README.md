# PC-DFC

Linking predictive coding and deep feedback control

## Installation

1. Install [uv](https://docs.astral.sh/uv/#installation)
2. Install Python dependencies with `uv sync --locked`

## Running code

To run any command within the environment, preface the command with `uv run`:
```
uv run python main.py
```

### Using the cluster

On the Slurm cluster, use `sbatch lauch-slurm.job` instead of `uv run`:
```
sbatch launch-slurm.job python main.py
```

### Sweeps on the cluster

To run a new sweep, create a YAML file under `sweeps/`, then do:
```
sbatch launch-sweep.job sweeps/<your yaml file> <runs per GPU job>
```

This creates the sweep once, then submits enough GPU-array jobs to cover it.
Each GPU job runs up to `<runs per GPU job>` W&B agents concurrently; the
final job runs only the remainder. For example,
`sbatch launch-sweep.job sweeps/my-sweep.yaml 4` runs up to four sweep
configurations concurrently on each allocated GPU. Choose the parallelism
based on per-run GPU memory use. Each agent's output is written to its own
file under `slurm-logs/`.

To continue a timed-out sweep from the latest checkpoint of each prior run,
pass its W&B sweep ID in place of the YAML file:
```
sbatch launch-sweep.job <prior-sweep-id> <runs per GPU job>
```
Runs without a readable checkpoint are skipped and reported during setup. The
continuation creates a new W&B sweep; its run-to-checkpoint manifest is stored
under the Git-ignored `slurm-logs/continuations/` directory.
