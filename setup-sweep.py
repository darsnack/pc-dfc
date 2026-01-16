import sys
import yaml
import wandb
from pathlib import Path

def main(sweep_file: Path):
    # read sweep yaml
    with open(sweep_file, "r") as f:
        sweep_config = yaml.safe_load(f)
    # get sweep id
    sweep_id = wandb.sweep(sweep_config, entity="mitra-ryan-kyle", project="pc-dfc")
    # get number of runs
    api = wandb.Api()
    sweep = api.sweep(f"mitra-ryan-kyle/pc-dfc/{sweep_id}")
    nruns = sweep.expected_run_count
    # print results
    print(f"{sweep_id} {nruns}")

if __name__ == "__main__":
    main(Path(sys.argv[1]))
