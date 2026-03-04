import os
import hydra
from hydra.core.hydra_config import HydraConfig
import wandb
import jax
import jax.numpy as jnp
import jax.random as jrng
import orbax.checkpoint as ocp
from flax import nnx
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from projectlib.models import MLP, Activation
from projectlib.dynamics import pc_step, run
from projectlib.data import load_dataset, build_dataloader, default_data_transforms
from projectlib.utils import noop, accuracy, get_model_params, compute_param_change_norm

def save_checkpoint(checkpoint_manager, model, epoch):
    _, state = nnx.split(model)
    checkpoint_manager.save(epoch, {"model": state})
    checkpoint_manager.wait_until_finished()

def load_checkpoint(checkpoint_path, model):
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path,
        item_handlers=ocp.StandardCheckpointHandler(),
        options=ocp.CheckpointManagerOptions(create=False)
    )
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        _, abstract_state = nnx.split(model)
        restored = checkpoint_manager.restore(latest_step, {"model": abstract_state})
        checkpoint_manager.wait_until_finished()
        nnx.update(model, restored["model"])
        return model, latest_step
    return model, None

def run_sample(cfg, rngs, x, y, model, epoch, sample_idx):
    # Store parameters after the nudge
    pre_nudge_params = get_model_params(model)
    if cfg.dynamics.sync_updates:
        _, _, timesteps = run(model, pc_step, x,
                              disable_tqdm=True,
                              tau=cfg.dynamics.time_constant,
                              eta=cfg.training.learning_rate,
                              min_timesteps=cfg.dynamics.min_timesteps,
                              max_timesteps=cfg.dynamics.max_timesteps,
                              tolerance=cfg.dynamics.convergence_tolerance,
                              tolerance_style=cfg.dynamics.tolerance_style,
                              learn=True,
                              rclamp=y)
    else:
        _, _, timesteps = run(model, pc_step, x,
                              disable_tqdm=True,
                              tau=cfg.dynamics.time_constant,
                              eta=cfg.training.learning_rate,
                              min_timesteps=20,
                              max_timesteps=20,
                              tolerance=cfg.dynamics.convergence_tolerance,
                              tolerance_style=cfg.dynamics.tolerance_style,
                              learn=False,
                              rclamp=y)
        _, _, timesteps = run(model, pc_step, x,
                              disable_tqdm=True,
                              tau=100000,
                              eta=cfg.training.learning_rate,
                              min_timesteps=1,
                              max_timesteps=1,
                              tolerance=cfg.dynamics.convergence_tolerance,
                              tolerance_style=cfg.dynamics.tolerance_style,
                              learn=True,
                              rclamp=y)
    error_val, activations = test_sample(cfg, rngs, model, x, y)
    # Store parameters after the nudge
    post_nudge_params = get_model_params(model)
    # Calculate parameter change norm
    param_change_norm = compute_param_change_norm(pre_nudge_params, post_nudge_params)

    wandb.log({
        "epoch": epoch,
        "train/sample": sample_idx,
        "train/timesteps": timesteps,
        "train/error": error_val,
        "train/weight_change_norm": float(param_change_norm)
    }, commit=True)

    return activations

def initialize_to_fp(cfg, model, sample):
    _, activations, init_steps = run(model, pc_step, sample,
                                     disable_tqdm=True,
                                     tau=cfg.dynamics.time_constant,
                                     tolerance_style=cfg.dynamics.tolerance_style,
                                     learn=False)
    tqdm.write(f"Initial convergence took {init_steps} timesteps")
    wandb.log({"initial_convergence_timesteps": init_steps})
    return activations

def test_sample(cfg, rngs, model, x, y):
    init_activations = model.get_activations()
    fp_input = jrng.normal(rngs.train(), x.shape) * 1e-1
    _ = initialize_to_fp(cfg, model, fp_input)
    _, activations, _ = run(model, pc_step, x,
                            disable_tqdm=True,
                            tau=cfg.dynamics.time_constant,
                            eta=cfg.training.learning_rate,
                            min_timesteps=cfg.dynamics.min_timesteps,
                            max_timesteps=cfg.dynamics.max_timesteps,
                            tolerance=cfg.dynamics.convergence_tolerance,
                            tolerance_style=cfg.dynamics.tolerance_style,
                            learn=False)
    error_val = float(jnp.mean(jnp.linalg.norm(activations[-1][:y.shape[0]] - y, axis=-1)))
    model.set_activations(init_activations)

    return error_val, activations

def train_epoch(cfg, rngs, model, dataset_trn, epoch):
    epoch_errors = []
    epoch_accuracies = []

    for sample_idx, (x, y) in tqdm(enumerate(dataset_trn), desc="training", position=1, leave=False):
        global_sample_idx = sample_idx + len(dataset_trn) * epoch
        x, y = jnp.array(x), jnp.array(y)
        v_fp = initialize_to_fp(cfg, model, x)
        v_fp = run_sample(cfg, rngs, x, y, model, epoch, global_sample_idx)

        epoch_errors.append(float(jnp.mean(jnp.linalg.norm(v_fp[-1][:y.shape[0]] - y, axis=-1))))
        epoch_accuracies.append(float(accuracy(v_fp[-1][:y.shape[0]], y)))

        wandb.log({
            "epoch": epoch,
            "train/sample": global_sample_idx,
            "train/batch_final_error": epoch_errors[-1],
            "train/batch_final_accuracy": epoch_accuracies[-1]
        }, commit=(sample_idx < len(dataset_trn) - 1))

    wandb.log({"epoch": epoch,
               "train/avg_epoch_error": jnp.mean(jnp.array(epoch_errors)),
               "train/avg_epoch_accuracy": jnp.mean(jnp.array(epoch_accuracies))})

def test_epoch(cfg, model, v_fp, dataset_tst, epoch, log_group = "test"):
    epoch_errors = []
    epoch_accuracies = []
    for sample_idx, (x, y) in tqdm(enumerate(dataset_tst), desc="testing", position=1, leave=False):
        global_sample_idx = sample_idx + len(dataset_tst) * epoch
        x, y = jnp.array(x), jnp.array(y)
        model.set_activations(v_fp)

        _, activations, timesteps = run(model, pc_step, x,
                                        disable_tqdm=True,
                                        tau=cfg.dynamics.time_constant,
                                        min_timesteps=cfg.dynamics.min_timesteps,
                                        max_timesteps=cfg.dynamics.max_timesteps,
                                        tolerance=cfg.dynamics.convergence_tolerance,
                                        tolerance_style=cfg.dynamics.tolerance_style,
                                        learn=False)

        v_fp = activations
        error_val = float(jnp.mean(jnp.linalg.norm(v_fp[-1][:y.shape[0]] - y, axis=-1)))
        epoch_errors.append(error_val)
        epoch_accuracies.append(float(accuracy(v_fp[-1][:y.shape[0]], y)))

        wandb.log({
            "epoch": epoch,
            f"{log_group}/sample": global_sample_idx,
            f"{log_group}/total_timesteps_per_sample": timesteps,
            f"{log_group}/batch_final_error": error_val,
            f"{log_group}/batch_final_accuracy": epoch_accuracies[-1]
        }, commit=(sample_idx < len(dataset_tst) - 1))

    wandb.log({"epoch": epoch,
               f"{log_group}/avg_epoch_error": jnp.mean(jnp.array(epoch_errors)),
               f"{log_group}/avg_epoch_accuracy": jnp.mean(jnp.array(epoch_accuracies))})

@hydra.main(config_path="./configs", config_name="main", version_base=None)
def main(cfg):
    rngs = nnx.Rngs(cfg.seed,
                    params=cfg.seed + 1,
                    train=cfg.seed + 2,
                    data=cfg.seed + 3,
                    activations=cfg.seed + 4)

    if cfg.model.activation_fn == "noop":
        activation_fn = noop
    elif cfg.model.activation_fn == "tanh":
        activation_fn = nnx.tanh
    else:
        raise ValueError(f"Unknown activation fn: {cfg.model.activation_fn}")
    if cfg.training.batch_size >= 1:
        with nnx.split_rngs(rngs, splits=cfg.training.batch_size, only="activations"):
            state_axes = nnx.StateAxes({(Activation, "activations"): 0, ...: None})
            model = nnx.vmap(lambda r:
                MLP(cfg.model.layer_dims,
                    activation_fn=activation_fn,
                    kernel_init_scale=cfg.model.get("kernel_init_scale", 1.0),
                    rngs=r),
            in_axes=(state_axes,), out_axes=state_axes)(rngs)
    else:
        model = MLP(cfg.model.layer_dims,
                    activation_fn=activation_fn,
                    kernel_init_scale=cfg.model.get("kernel_init_scale", 1.0),
                    rngs=rngs)

    transforms = default_data_transforms(cfg.data)
    data = load_dataset(cfg.data, rngs.data(), transforms)
    input_shape = data["test"][0][0].shape
    trn_loader = build_dataloader(data["train"], batch_size=cfg.training.batch_size, drop_remainder=True)
    tst_loader = build_dataloader(data["test"], batch_size=cfg.training.batch_size, drop_remainder=True)

    wandb.init(entity="mitra-ryan-kyle",
               project="pc-dfc",
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    hydra_output_dir = HydraConfig.get().runtime.output_dir
    checkpoint_dir = os.path.join(hydra_output_dir, "checkpoints")
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_dir,
        item_handlers=ocp.StandardCheckpointHandler(),
        options=ocp.CheckpointManagerOptions(create=True, max_to_keep=5)
    )

    if cfg.checkpointing.load_from is not None:
        model, loaded_epoch = load_checkpoint(cfg.checkpointing.load_from, model)
        if loaded_epoch is not None:
            tqdm.write(f"Loaded checkpoint from epoch {loaded_epoch}")

    fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
    init_v_fp = initialize_to_fp(cfg, model, fp_input)
    test_epoch(cfg, model, init_v_fp, trn_loader, 0, log_group="train")
    test_epoch(cfg, model, init_v_fp, tst_loader, 0)
    if cfg.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=False):
            for epoch in tqdm(range(1, cfg.training.epochs + 1), desc="epochs", position=0, leave=True):
                train_epoch(cfg, rngs, model, trn_loader, epoch)
                fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
                init_v_fp = initialize_to_fp(cfg, model, fp_input)
                test_epoch(cfg, model, init_v_fp, tst_loader, epoch)
                save_checkpoint(checkpoint_manager, model, epoch)
    else:
        for epoch in tqdm(range(1, cfg.training.epochs + 1), desc="epochs", position=0, leave=True):
            train_epoch(cfg, rngs, model, trn_loader, epoch)
            fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
            init_v_fp = initialize_to_fp(cfg, model, fp_input)
            test_epoch(cfg, model, init_v_fp, tst_loader, epoch)
            save_checkpoint(checkpoint_manager, model, epoch)


if __name__ == "__main__":
    main()
