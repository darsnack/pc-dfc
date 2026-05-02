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
from projectlib.utils import noop, get_model_params, compute_param_change_norm

def make_step_fn(nmodels):
    if nmodels <= 1:
        return pc_step

    model_axes = nnx.StateAxes({...: 0})

    def step(model, x, **kwargs):
        rclamp = kwargs.get('rclamp', None)
        tau = kwargs.get('tau', 1)
        eta = kwargs.get('eta', 1)
        learn = kwargs.get('learn', False)
        if rclamp is not None:
            vmapped = nnx.vmap(
                lambda m, x, rc, tau, eta: pc_step(m, x, tau=tau, eta=eta, rclamp=rc, learn=learn),
                in_axes=(model_axes, None, 0, None, None),
                out_axes=0,
            )
            return vmapped(model, x, rclamp, tau, eta)
        else:
            vmapped = nnx.vmap(
                lambda m, x, rc, tau, eta: pc_step(m, x, tau=tau, eta=eta, rclamp=None, learn=learn),
                in_axes=(model_axes, None, None, None, None),
                out_axes=0,
            )
            return vmapped(model, x, rclamp, tau, eta)

    return step

def _create_single_model(rngs, cfg, activation_fn):
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
    return model

def save_checkpoint(checkpoint_manager, model, epoch):
    _, state = nnx.split(model)
    checkpoint_manager.save(epoch, {"model": state})
    checkpoint_manager.wait_until_finished()

def load_checkpoint(checkpoint_path, model, sample_input=None):
    checkpoint_manager = ocp.CheckpointManager(
        checkpoint_path,
        item_handlers=ocp.StandardCheckpointHandler(),
        options=ocp.CheckpointManagerOptions(create=False)
    )
    latest_step = checkpoint_manager.latest_step()
    if latest_step is not None:
        if sample_input is not None:
            _ = model(sample_input)
        _, abstract_state = nnx.split(model)
        restored = checkpoint_manager.restore(latest_step, {"model": abstract_state})
        checkpoint_manager.wait_until_finished()
        nnx.update(model, restored["model"])
        return model, latest_step
    return model, None

def run_sample(cfg, rngs, x, y, model, step_fn, epoch, sample_idx, integration_step):
    nmodels = cfg.nmodels
    pre_nudge_params = get_model_params(model)

    rclamp = y
    if nmodels > 1:
        rclamp = jnp.broadcast_to(y, (nmodels, *y.shape))

    if cfg.dynamics.sync_updates:
        _, _, timesteps = run(model, step_fn, x,
                              nmodels=nmodels,
                              disable_tqdm=True,
                              tau=cfg.dynamics.time_constant,
                              eta=cfg.training.learning_rate,
                              min_timesteps=cfg.dynamics.min_timesteps,
                              max_timesteps=cfg.dynamics.max_timesteps,
                              tolerance=cfg.dynamics.convergence_tolerance,
                              tolerance_style=cfg.dynamics.tolerance_style,
                              learn=True,
                              rclamp=rclamp)
        total_timesteps = timesteps
    else:
        _, _, timesteps_inference = run(model, step_fn, x,
                                        nmodels=nmodels,
                                        disable_tqdm=True,
                                        tau=cfg.dynamics.time_constant,
                                        eta=cfg.training.learning_rate,
                                        min_timesteps=20,
                                        max_timesteps=20,
                                        tolerance=cfg.dynamics.convergence_tolerance,
                                        tolerance_style=cfg.dynamics.tolerance_style,
                                        learn=False,
                                        rclamp=rclamp)
        _, _, timesteps_learning = run(model, step_fn, x,
                                       nmodels=nmodels,
                                       disable_tqdm=True,
                                       tau=100000,
                                       eta=cfg.training.learning_rate,
                                       min_timesteps=1,
                                       max_timesteps=1,
                                       tolerance=cfg.dynamics.convergence_tolerance,
                                       tolerance_style=cfg.dynamics.tolerance_style,
                                       learn=True,
                                       rclamp=rclamp)
        total_timesteps = timesteps_inference + timesteps_learning

    integration_step += total_timesteps

    error_val, activations = test_sample(cfg, rngs, model, step_fn, x, y)
    post_nudge_params = get_model_params(model)
    param_change_norm = compute_param_change_norm(pre_nudge_params, post_nudge_params)

    wandb.log({
        "epoch": epoch,
        "train/sample": sample_idx,
        "train/timesteps": total_timesteps,
        "train/integration_step": integration_step,
        "train/error": error_val,
        "train/weight_change_norm": param_change_norm
    }, commit=True)

    return activations, total_timesteps, integration_step

def runs_with_nudges(cfg, rngs, x, y, model, step_fn, v_fp, epoch, sample_idx,
                     integration_step):
    nmodels = cfg.nmodels
    nudge_count = 0
    total_timesteps = 0

    def _nudge_converged(v_fp_last, y, tol):
        dists = jnp.linalg.norm(v_fp_last - y, axis=-1)
        if nmodels > 1:
            return jnp.max(jnp.median(dists, axis=-1)) <= tol
        else:
            return jnp.median(dists) <= tol

    while not _nudge_converged(v_fp[-1], y, cfg.nudge.convergence_tolerance):
        if nudge_count > cfg.nudge.max_count:
            break
        pre_nudge_params = get_model_params(model)
        nudge_count += 1
        rclamp = v_fp[-1] + (y - v_fp[-1]) * cfg.nudge.scaling_factor

        if cfg.dynamics.sync_updates:
            _, activations, timesteps = run(model, step_fn, x,
                                            nmodels=nmodels,
                                            disable_tqdm=True,
                                            tau=cfg.dynamics.time_constant,
                                            eta=cfg.training.learning_rate,
                                            min_timesteps=cfg.dynamics.min_timesteps,
                                            max_timesteps=cfg.dynamics.max_timesteps,
                                            tolerance=cfg.dynamics.convergence_tolerance,
                                            tolerance_style=cfg.dynamics.tolerance_style,
                                            learn=True,
                                            rclamp=rclamp)
            nudge_timesteps = timesteps
        else:
            _, _, timesteps_inference = run(model, step_fn, x,
                                            nmodels=nmodels,
                                            disable_tqdm=True,
                                            tau=cfg.dynamics.time_constant,
                                            eta=cfg.training.learning_rate,
                                            min_timesteps=20,
                                            max_timesteps=20,
                                            tolerance=cfg.dynamics.convergence_tolerance,
                                            tolerance_style=cfg.dynamics.tolerance_style,
                                            learn=False,
                                            rclamp=rclamp)
            _, activations, timesteps_learning = run(model, step_fn, x,
                                                    nmodels=nmodels,
                                                    disable_tqdm=True,
                                                    tau=100000,
                                                    eta=cfg.training.learning_rate,
                                                    min_timesteps=1,
                                                    max_timesteps=1,
                                                    tolerance=cfg.dynamics.convergence_tolerance,
                                                    tolerance_style=cfg.dynamics.tolerance_style,
                                                    learn=True,
                                                    rclamp=rclamp)
            nudge_timesteps = timesteps_inference + timesteps_learning

        v_fp = activations
        total_timesteps += nudge_timesteps
        integration_step += nudge_timesteps

        post_nudge_params = get_model_params(model)
        param_change_norm = compute_param_change_norm(pre_nudge_params, post_nudge_params)

        error_val = jnp.mean(jnp.linalg.norm(v_fp[-1] - y, axis=-1), axis=-1)

        log_dict = {
            "epoch": epoch,
            "train/sample": sample_idx,
            "train/nudge": nudge_count,
            "train/timesteps_per_nudge": nudge_timesteps,
            "train/integration_step": integration_step,
            "train/dist_to_target": error_val,
            "train/weight_change_norm": param_change_norm
        }
        wandb.log(log_dict, commit=bool(jnp.all(error_val > cfg.nudge.convergence_tolerance)))

        clamp_mean = jnp.mean(rclamp, axis=0)
        act_mean   = jnp.mean(activations[-1], axis=0)

        log_dict = {}
        for u in range(clamp_mean.shape[0]):
            log_dict[f"activity/u{u}/clamp"]    = float(clamp_mean[u])
            log_dict[f"activity/u{u}/activity"] = float(act_mean[u])
        wandb.log(log_dict)

    return nudge_count, total_timesteps, v_fp, integration_step

def initialize_to_fp(cfg, model, step_fn, sample):
    _, activations, init_steps = run(model, step_fn, sample,
                                     nmodels=cfg.nmodels,
                                     disable_tqdm=True,
                                     tau=cfg.dynamics.time_constant,
                                     tolerance_style=cfg.dynamics.tolerance_style,
                                     learn=False)
    tqdm.write(f"Initial convergence took {init_steps} timesteps")
    wandb.log({"initial_convergence_timesteps": init_steps})
    return activations

def test_sample(cfg, rngs, model, step_fn, x, y):
    init_activations = model.get_activations()
    fp_input = jrng.normal(rngs.train(), x.shape) * 1e-1
    _ = initialize_to_fp(cfg, model, step_fn, fp_input)
    _, activations, _ = run(model, step_fn, x,
                            nmodels=cfg.nmodels,
                            disable_tqdm=True,
                            tau=cfg.dynamics.time_constant,
                            eta=cfg.training.learning_rate,
                            min_timesteps=cfg.dynamics.min_timesteps,
                            max_timesteps=cfg.dynamics.max_timesteps,
                            tolerance=cfg.dynamics.convergence_tolerance,
                            tolerance_style=cfg.dynamics.tolerance_style,
                            learn=False)
    error_val = jnp.mean(jnp.linalg.norm(activations[-1] - y, axis=-1), axis=-1)
    model.set_activations(init_activations)

    return error_val, activations

def _per_model_accuracy(pred, y):
    return jnp.mean(jnp.argmax(pred, axis=-1) == jnp.argmax(y, axis=-1), axis=-1)

def train_epoch(cfg, rngs, model, step_fn, dataset_trn, epoch, integration_step):
    epoch_errors = []
    epoch_accuracies = []
    epoch_nudge_counts = []

    for sample_idx, (x, y) in tqdm(enumerate(dataset_trn), desc="training", position=1, leave=False):
        global_sample_idx = sample_idx + len(dataset_trn) * epoch
        x, y = jnp.array(x), jnp.array(y)
        v_fp = initialize_to_fp(cfg, model, step_fn, x)

        if cfg.nudge.enabled:
            nudge_count, total_timesteps, v_fp, integration_step = runs_with_nudges(
                cfg, rngs, x, y, model, step_fn, v_fp, epoch, global_sample_idx, integration_step,
            )
            error_val, test_activations = test_sample(cfg, rngs, model, step_fn, x, y)
            epoch_nudge_counts.append(nudge_count)
            epoch_errors.append(error_val)
            epoch_accuracies.append(_per_model_accuracy(test_activations[-1], y))
        else:
            v_fp, total_timesteps, integration_step = run_sample(
                cfg, rngs, x, y, model, step_fn, epoch, global_sample_idx, integration_step,
            )
            epoch_errors.append(jnp.mean(jnp.linalg.norm(v_fp[-1] - y, axis=-1), axis=-1))
            epoch_accuracies.append(_per_model_accuracy(v_fp[-1], y))

        log_dict = {
            "epoch": epoch,
            "train/sample": global_sample_idx,
            "train/total_timesteps_per_sample": total_timesteps,
            "train/integration_step": integration_step,
            "train/batch_final_error": epoch_errors[-1],
            "train/batch_final_accuracy": epoch_accuracies[-1]
        }
        if cfg.nudge.enabled:
            log_dict["train/total_nudges_per_sample"] = nudge_count
        wandb.log(log_dict, commit=(sample_idx < len(dataset_trn) - 1))

    wandb.log({"epoch": epoch,
               "train/avg_epoch_error": jnp.mean(jnp.array(epoch_errors), axis=0),
               "train/avg_epoch_accuracy": jnp.mean(jnp.array(epoch_accuracies), axis=0)})

    return integration_step

def test_epoch(cfg, model, step_fn, v_fp, dataset_tst, epoch, log_group = "test"):
    epoch_errors = []
    epoch_accuracies = []
    for sample_idx, (x, y) in tqdm(enumerate(dataset_tst), desc="testing", position=1, leave=False):
        global_sample_idx = sample_idx + len(dataset_tst) * epoch
        x, y = jnp.array(x), jnp.array(y)
        model.set_activations(v_fp)

        _, activations, timesteps = run(model, step_fn, x,
                                        nmodels=cfg.nmodels,
                                        disable_tqdm=True,
                                        tau=cfg.dynamics.time_constant,
                                        min_timesteps=cfg.dynamics.min_timesteps,
                                        max_timesteps=cfg.dynamics.max_timesteps,
                                        tolerance=cfg.dynamics.convergence_tolerance,
                                        tolerance_style=cfg.dynamics.tolerance_style,
                                        learn=False)

        error_val = jnp.mean(jnp.linalg.norm(activations[-1] - y, axis=-1), axis=-1)
        epoch_errors.append(error_val)
        epoch_accuracies.append(_per_model_accuracy(activations[-1], y))

        wandb.log({
            "epoch": epoch,
            f"{log_group}/sample": global_sample_idx,
            f"{log_group}/total_timesteps_per_sample": timesteps,
            f"{log_group}/batch_final_error": epoch_errors[-1],
            f"{log_group}/batch_final_accuracy": epoch_accuracies[-1]
        }, commit=(sample_idx < len(dataset_tst) - 1))

    wandb.log({"epoch": epoch,
               f"{log_group}/avg_epoch_error": jnp.mean(jnp.array(epoch_errors), axis=0),
               f"{log_group}/avg_epoch_accuracy": jnp.mean(jnp.array(epoch_accuracies), axis=0)})

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

    if cfg.nmodels > 1:
        with nnx.split_rngs(rngs, splits=cfg.nmodels, only="params"):
            nmodels_axes = nnx.StateAxes({...: 0})
            model = nnx.vmap(
                lambda r: _create_single_model(r, cfg, activation_fn),
                in_axes=(nmodels_axes,), out_axes=nmodels_axes,
            )(rngs)
    else:
        model = _create_single_model(rngs, cfg, activation_fn)

    step_fn = make_step_fn(cfg.nmodels)

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
        ckpt_sample_input = jnp.zeros((cfg.training.batch_size, *input_shape), dtype=jnp.float32)
        model, loaded_epoch = load_checkpoint(cfg.checkpointing.load_from, model, ckpt_sample_input)
        if loaded_epoch is not None:
            tqdm.write(f"Loaded checkpoint from epoch {loaded_epoch}")

    integration_step = 0
    fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
    init_v_fp = initialize_to_fp(cfg, model, step_fn, fp_input)
    test_epoch(cfg, model, step_fn, init_v_fp, trn_loader, 0, log_group="train")
    test_epoch(cfg, model, step_fn, init_v_fp, tst_loader, 0)
    if cfg.profile:
        with jax.profiler.trace("/tmp/jax-trace", create_perfetto_trace=False):
            for epoch in tqdm(range(1, cfg.training.epochs + 1), desc="epochs", position=0, leave=True):
                integration_step = train_epoch(
                    cfg,
                    rngs,
                    model,
                    step_fn,
                    trn_loader,
                    epoch,
                    integration_step,
                )
                fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
                init_v_fp = initialize_to_fp(cfg, model, step_fn, fp_input)
                test_epoch(cfg, model, step_fn, init_v_fp, tst_loader, epoch)
                save_checkpoint(checkpoint_manager, model, epoch)
    else:
        for epoch in tqdm(range(1, cfg.training.epochs + 1), desc="epochs", position=0, leave=True):
            integration_step = train_epoch(
                cfg,
                rngs,
                model,
                step_fn,
                trn_loader,
                epoch,
                integration_step,
            )
            fp_input = jrng.normal(rngs.train(), (cfg.training.batch_size, *input_shape)) * 1e-1
            init_v_fp = initialize_to_fp(cfg, model, step_fn, fp_input)
            test_epoch(cfg, model, step_fn, init_v_fp, tst_loader, epoch)
            save_checkpoint(checkpoint_manager, model, epoch)


if __name__ == "__main__":
    main()
