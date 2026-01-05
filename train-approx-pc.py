import hydra
import wandb
import jax
import jax.numpy as jnp
import jax.random as jrng
from flax import nnx
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from projectlib.models import MLP
from projectlib.dynamics import pc_step, current_pc_vjp, approx_pc_step, run
from projectlib.data import load_dataset, build_dataloader, default_data_transforms
from projectlib.utils import noop, get_model_params, compute_param_change_norm

def runs_with_nudges(cfg, x, y, model, v_fp, vjps, epoch, sample_idx):
    # Keep nudging toward current sample until convergence
    nudge_count = 0
    total_timesteps = 0
    while jnp.median(jnp.linalg.norm(v_fp[-1][:y.shape[0]] - y, axis=-1)) > cfg.nudge.convergence_tolerance:
        if nudge_count > cfg.nudge.max_count:
            # print("Nudge count exceeded maximum")
            break
        # Store parameters before the nudge
        pre_nudge_params = get_model_params(model)
        nudge_count += 1
        # Compute nudge in direction of target/label
        rclamp = v_fp[-1][:y.shape[0]] + (y - v_fp[-1][:y.shape[0]]) * cfg.nudge.scaling_factor
        # Run dynamics until convergence
        _, activations, timesteps = run(model, approx_pc_step, x,
                                        disable_tqdm=True,
                                        vjps=vjps,
                                        tau=cfg.dynamics.time_constant,
                                        eta=cfg.training.learning_rate,
                                        max_timesteps=cfg.dynamics.max_timesteps,
                                        tolerance=cfg.dynamics.convergence_tolerance,
                                        tolerance_style=cfg.dynamics.tolerance_style,
                                        learn=True,
                                        rclamp=rclamp)
        # Update fixed point
        v_fp = activations
        vjps = current_pc_vjp(model, x)

        # Store parameters after the nudge
        post_nudge_params = get_model_params(model)
        # Calculate parameter change norm
        param_change_norm = compute_param_change_norm(pre_nudge_params, post_nudge_params)

        # Record nudge statistics
        error_val = float(jnp.mean(jnp.linalg.norm(v_fp[-1][:y.shape[0]] - y, axis=-1)))
        total_timesteps += timesteps

        # Log each nudge step to wandb
        wandb.log({
            "epoch": epoch,
            "train/sample": sample_idx,
            "train/nudge": nudge_count,
            "train/timesteps_per_nudge": timesteps,
            "train/error": error_val,
            "train/weight_change_norm": float(param_change_norm)
        }, commit=(error_val > cfg.nudge.convergence_tolerance))

    return nudge_count, total_timesteps, v_fp, vjps

def initialize_to_fp(cfg, model, sample):
    _, activations, init_steps = run(model, pc_step, sample,
                                     disable_tqdm=True,
                                     tau=cfg.dynamics.time_constant,
                                     tolerance_style=cfg.dynamics.tolerance_style,
                                     learn=False)
    print(f"Initial convergence took {init_steps} timesteps")
    # Log initial convergence to wandb
    wandb.log({"initial_convergence_timesteps": init_steps})
    # Store fixed point of last layer
    init_v_fp = activations
    init_vjps = current_pc_vjp(model, sample)

    return init_v_fp, init_vjps

def train_epoch(cfg, model, v_fp, vjps, dataset_trn, epoch):
    epoch_nudge_counts = []
    epoch_timestep_counts = []
    epoch_errors = []

    for sample_idx, (x, y) in tqdm(enumerate(dataset_trn), desc="training", position=1, leave=False):
        x, y = jnp.array(x), jnp.array(y)
        nudge_count, total_timesteps, v_fp, vjps = runs_with_nudges(cfg, x, y, model, v_fp, vjps, epoch, sample_idx)

        # Log summary metrics for this sample
        epoch_nudge_counts.append(nudge_count)
        epoch_timestep_counts.append(total_timesteps)
        epoch_errors.append(float(jnp.mean(jnp.linalg.norm(v_fp[-1][:y.shape[0]] - y, axis=-1))))

        # Log sample-level metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/sample": sample_idx,
            "train/total_nudges_per_sample": nudge_count,
            "train/total_timesteps_per_sample": total_timesteps,
            "train/batch_final_error": epoch_errors[-1]
        }, commit=(sample_idx < len(dataset_trn) - 1))

    wandb.log({"epoch": epoch, "train/avg_epoch_error": jnp.mean(jnp.array(epoch_errors))})

def test_epoch(cfg, model, v_fp, vjps, dataset_tst, epoch):
    epoch_errors = []
    for sample_idx, (x, y) in tqdm(enumerate(dataset_tst), desc="testing", position=1, leave=False):
        x, y = jnp.array(x), jnp.array(y)
        # Set activation to fixed point
        model.set_activations(v_fp)

        # Run dynamics until convergence
        _, activations, timesteps = run(model, approx_pc_step, x,
                                        disable_tqdm=True,
                                        vjps=vjps,
                                        tau=cfg.dynamics.time_constant,
                                        learn=False)

        # Record nudge statistics
        v_fp = activations
        error_val = float(jnp.mean(jnp.linalg.norm(v_fp[-1][:y.shape[0]] - y, axis=-1)))
        epoch_errors.append(error_val)

        # Log sample-level metrics to wandb
        wandb.log({
            "epoch": epoch,
            "test/sample": sample_idx,
            "test/total_timesteps_per_sample": timesteps,
            "test/batch_final_error": error_val
        }, commit=(sample_idx < len(dataset_tst) - 1))

    wandb.log({"epoch": epoch, "test/avg_epoch_error": jnp.mean(jnp.array(epoch_errors))})

@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(cfg):
    # Initialize random number generator
    rngs = nnx.Rngs(cfg.seed, params=cfg.seed + 1, train=cfg.seed + 2, data=cfg.seed + 3)

    # Build model
    if cfg.model.activation_fn == "noop":
        activation_fn = noop
    elif cfg.model.activation_fn == "tanh":
        activation_fn = nnx.tanh
    else:
        raise ValueError(f"Unknown activation fn: {cfg.model.activation_fn}")
    model = MLP(cfg.model.layer_dims,
                activation_fn=activation_fn,
                batch_size=cfg.training.batch_size,
                rngs=rngs)

    # Load datasets
    transforms = default_data_transforms(cfg.data)
    data = load_dataset(cfg.data, rngs.data(), transforms)
    input_shape = data["test"][0][0].shape
    trn_loader = build_dataloader(data["train"], batch_size=cfg.training.batch_size, drop_remainder=True)
    tst_loader = build_dataloader(data["test"], batch_size=cfg.training.batch_size, drop_remainder=True)

    # Initialize wandb
    wandb.init(entity="mitra-ryan-kyle", project="pc-dfc", config=OmegaConf.to_container(cfg))

    if cfg.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
            for epoch in tqdm(range(cfg.training.epochs), desc="epochs", position=0, leave=True):
                # Initialize to a fixed point using random Gaussian input
                # (think about this deeper later)
                fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
                init_v_fp, init_vjps = initialize_to_fp(cfg, model, fp_input)
                train_epoch(cfg, model, init_v_fp, init_vjps, trn_loader, epoch)
                fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
                init_v_fp, init_vjps = initialize_to_fp(cfg, model, fp_input)
                test_epoch(cfg, model, init_v_fp, init_vjps, tst_loader, epoch)
    else:
        for epoch in tqdm(range(cfg.training.epochs), desc="epochs", position=0, leave=True):
            # Initialize to a fixed point using random Gaussian input
            # (think about this deeper later)
            fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
            init_v_fp, init_vjps = initialize_to_fp(cfg, model, fp_input)
            train_epoch(cfg, model, init_v_fp, init_vjps, trn_loader, epoch)
            fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
            init_v_fp, init_vjps = initialize_to_fp(cfg, model, fp_input)
            test_epoch(cfg, model, init_v_fp, init_vjps, tst_loader, epoch)

    # # Log epoch-level summary statistics
    # wandb.log({
    #     "epoch": epoch,
    #     "avg_nudges_per_sample": sum(epoch_nudge_counts) / len(epoch_nudge_counts),
    #     "max_nudges": max(epoch_nudge_counts),
    #     "avg_timesteps_per_sample": sum(epoch_timestep_counts) / len(epoch_timestep_counts),
    #     "max_timesteps": max(epoch_timestep_counts),
    #     "avg_final_error": sum(epoch_errors) / len(epoch_errors)
    # })

    # wandb.finish()


if __name__ == "__main__":
    main()
