import argparse
import glob
import os
import re
import subprocess
import sys


def parse_overrides_seed(overrides_path):
    with open(overrides_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("- seed="):
                return int(line.split("=", 1)[1])
            m = re.match(r"^-+\s*seed=(\d+)", line)
            if m:
                return int(m.group(1))
    return 0


def find_latest_checkpoint_dir(source_lr, source_seed):
    base = os.path.join("outputs", "approx_pc_learning_rate_sweep", f"lr_{source_lr}")
    if not os.path.isdir(base):
        print(f"Error: no output directory found at {base}", file=sys.stderr)
        sys.exit(1)

    overrides_files = glob.glob(os.path.join(base, "*", "*", ".hydra", "overrides.yaml"))
    if not overrides_files:
        print(f"Error: no runs found under {base}", file=sys.stderr)
        sys.exit(1)

    matching = []
    for ov_path in overrides_files:
        run_dir = os.path.dirname(os.path.dirname(ov_path))
        seed = parse_overrides_seed(ov_path)
        if seed == source_seed:
            matching.append(run_dir)

    if not matching:
        print(f"Error: no runs found for lr={source_lr}, seed={source_seed}", file=sys.stderr)
        sys.exit(1)

    matching.sort()
    latest_run = matching[-1]

    ckpt_dir = os.path.join(latest_run, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print(f"Error: no checkpoints directory in {latest_run}", file=sys.stderr)
        sys.exit(1)

    steps = [int(d) for d in os.listdir(ckpt_dir)
             if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit()]
    if not steps:
        print(f"Error: no checkpoint steps found in {ckpt_dir}", file=sys.stderr)
        sys.exit(1)

    latest_step = max(steps)
    print(f"Found checkpoint: {ckpt_dir} (latest step: {latest_step})")
    return ckpt_dir


def main():
    parser = argparse.ArgumentParser(description="PCN transplant experiment")
    parser.add_argument("--source-lr", type=float, required=True,
                        help="Approx PC learning rate to source checkpoint from")
    parser.add_argument("--source-seed", type=int, required=True,
                        help="Seed of the approx PC run to source checkpoint from")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command without executing")
    args = parser.parse_args()

    ckpt_dir = os.path.abspath(find_latest_checkpoint_dir(args.source_lr, args.source_seed))

    lr_str = str(args.source_lr)
    cmd = [
        "uv", "run", "train-pcn.py",
        f"checkpointing.load_from={ckpt_dir}",
        "training.epochs=1",
        f"training.learning_rate={lr_str}",
        f"seed={args.source_seed}",
        "data=mnist",
        "model.activation_fn=tanh",
        f"hydra.run.dir=./outputs/pcn_transplant/lr_{lr_str}/seed_{args.source_seed}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}",
    ]

    print(f"Command: {' '.join(cmd)}")

    if args.dry_run:
        print("(dry run — not executing)")
        return

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
