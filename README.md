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

## Reproduction

Below are the instructions for reproducing the figures in the paper.

### Figures 2–6

Each experiment saves its simulation data independently of plotting. Run these commands from the repository root; the explicit output directories keep the saved results easy to find.

We use the Jax CPU backend and 64-bit arithmetic for these experiments to ensure numerical accuracy. Set both environment variables before running the commands below:
```sh
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
```

Here is a summary of scripts associated with each figure:

| Figures | Experiment | Plotting |
| --- | --- | --- |
| 2–3: linear networks | `linear-experiment.py` | `linear-approx-plotting.py` |
| 4–5: nonlinear networks + relinearization | `nonlinear-experiment.py` | `nonlinear-plotting.py` |
| 6: nudging | `nudging-experiment.py` | `nudging-plotting.py` |

```sh
uv run python linear-experiment.py hydra.run.dir=outputs/linear
uv run python linear-approx-plotting.py experiment_dir=outputs/linear

uv run python nonlinear-experiment.py hydra.run.dir=outputs/nonlinear
uv run python nonlinear-plotting.py experiment_dir=outputs/nonlinear

uv run python nudging-experiment.py hydra.run.dir=outputs/nudging
uv run python nudging-plotting.py experiment_dir=outputs/nudging sf=0.1
uv run python nudging-plotting.py experiment_dir=outputs/nudging sf=0.9
```

The YAML files under `configs/` with the same names contain the experiment and plotting settings.
Plots are written alongside the saved data. Plotting can be rerun without simulating again.

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
