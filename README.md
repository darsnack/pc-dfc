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

To run a sweep, create a new YAML file under `sweeps/`, then do:
```
sbatch launch-sweep.job sweeps/<your yaml file>
```

That command keeps the existing behavior: it submits one GPU Slurm job for
each sweep run. To run several sweep configurations concurrently on one GPU
allocation instead, use:
```
sbatch launch-sweep-concurrent.job sweeps/<your yaml file> <max parallel runs>
```

For example, `sbatch launch-sweep-concurrent.job sweeps/my-sweep.yaml 4`
allocates one GPU and runs up to four W&B agents concurrently. Choose the
parallelism based on per-run GPU memory use. Omitting `<max parallel runs>`
starts all expected sweep runs at once. Each agent's output is written to its
own file under `slurm-logs/`.
