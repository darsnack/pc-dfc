import wandb
import jax.numpy as jnp
import jax.random as jrng
import jax.tree as jtu
from flax import nnx
from tqdm.auto import tqdm

from projectlib.models import MLP, Activation
from projectlib.dynamics import pc_step, current_pc_vjp, approx_pc_step, run
from projectlib.utils import noop, get_model_params, compute_param_change_norm


# Initialize random number generator
rngs = nnx.Rngs(0, params=1, train=2, data=3)

# Build model
layer_dims = [2, 2, 2, 1]
model = MLP(layer_dims, activation_fn=noop, rngs=rngs)

# Training parameters
num_epochs = 10
tau = 30
lr = 1e-3
nudge_eps = 1e-5
nudge_factor = 10

# Synthetic dataset
wtrue = jrng.normal(rngs.data(), (2, 1))
X = jrng.normal(rngs.data(), (10, 2)) * 1e-1
y = X @ wtrue
dataset_trn = [(x_i, y_i) for x_i, y_i in zip(X, y)]
X = jrng.normal(rngs.data(), (5, 2)) * 1e-1
y = X @ wtrue
dataset_tst = [(x_i, y_i) for x_i, y_i in zip(X, y)]

# Initialize wandb
wandb.init(
    project="pc-dfc",  # Project name
    config={
        "model_layers": layer_dims,
        "epochs": num_epochs,
        "tau": tau,
        "learning_rate": lr,
        "nudge_eps": nudge_eps,
        "nudge_factor": nudge_factor,
        "dataset_trn_size": len(dataset_trn),
        "dataset_tst_size": len(dataset_tst),
    }
)

# TODO: do we need to store v_fp separately?
#       (see pseudo-code uses model.activations[-1] directly)






def runs_with_nudges(x, y, model, v_fp, vjps, epoch, sample_idx):
    # Keep nudging toward current sample until convergence
    nudge_count = 0
    total_timesteps = 0
    while jnp.linalg.norm(v_fp - y) > nudge_eps:
        # Store parameters before the nudge
        pre_nudge_params = get_model_params(model)
        nudge_count += 1
        # Compute nudge in direction of target/label
        rclamp = v_fp + (y - v_fp) / nudge_factor
        # Run dynamics until convergence
        errors, activations, timesteps = run(model, approx_pc_step, x,
                                                disable_tqdm=True,
                                                vjps=vjps,
                                                tau=tau,
                                                eta=lr,
                                                learn=True,
                                                rclamp=rclamp)
        # Update fixed point
        v_fp = activations[-1][-1]
        vjps = current_pc_vjp(model, x)

        # Store parameters after the nudge
        post_nudge_params = get_model_params(model)
        # Calculate parameter change norm
        param_change_norm = compute_param_change_norm(pre_nudge_params, post_nudge_params)

        # Record nudge statistics
        error_val = float(jnp.linalg.norm(v_fp - y))
        total_timesteps += timesteps

        # Log each nudge step to wandb
        wandb.log({
            "epoch": epoch,
            "train/sample": sample_idx,
            "train/nudge": nudge_count,
            "train/timesteps_per_nudge": timesteps,
            "train/error": error_val,
            "train/weight_change_norm": float(param_change_norm)
        }, commit=(error_val > nudge_eps))

    return nudge_count, total_timesteps, v_fp, vjps



def train_epoch(config, model, dataset_trn, epoch):

    # Converge to fixed point initially on first example
    print("Initial convergence to fixed point:")
    errors, activations, init_steps = run(model, pc_step, dataset_trn[0][0],
                                        tau=tau, eta=lr, learn=False)
    print(f"Initial convergence took {init_steps} timesteps")
    # Log initial convergence to wandb
    wandb.log({"initial_convergence_timesteps": init_steps})

    # Store fixed point of last layer
    v_fp = activations[-1][-1]
    vjps = current_pc_vjp(model, dataset_trn[0][0])

    epoch_nudge_counts = []
    epoch_timestep_counts = []
    epoch_errors = []

    for sample_idx, (x, y) in enumerate(dataset_trn):
        nudge_count, total_timesteps, v_fp, vjps = runs_with_nudges(x, y, model, v_fp, vjps, epoch, sample_idx)

        # Log summary metrics for this sample
        epoch_nudge_counts.append(nudge_count)
        epoch_timestep_counts.append(total_timesteps)
        epoch_errors.append(float(jnp.linalg.norm(v_fp - y)))

        # Log sample-level metrics to wandb
        wandb.log({
            "epoch": epoch,
            "train/sample": sample_idx,
            "train/total_nudges_per_sample": nudge_count,
            "train/total_timesteps_per_sample": total_timesteps,
            "train/final_error": float(jnp.linalg.norm(v_fp - y))
        }, commit=(sample_idx < len(dataset_trn) - 1))


def test_epoch(config, model, dataset_test, epoch):

    # Update fixed point
    # TODO change input from first data point to random vector or random data point or smth
    errors, activations, init_steps = run(model, pc_step, dataset_tst[0][0],
                                        tau=tau, eta=lr, learn=False)
    v_fp_init = activations[-1][-1]
    vjps = current_pc_vjp(model, dataset_tst[0][0])

    for sample_idx, (x, y) in enumerate(dataset_test):

        model.set_activations(v_fp_init)
        
        # Run dynamics until convergence
        errors, activations, timesteps = run(model, approx_pc_step, x,
                                            disable_tqdm=True,
                                            vjps=vjps,
                                            tau=tau,
                                            eta=lr,
                                            learn=False)


        # Record nudge statistics
        v_fp = activations[-1][-1]
        error_val = float(jnp.linalg.norm(v_fp - y))

        # Log sample-level metrics to wandb
        wandb.log({
            "epoch": epoch,
            "test/sample": sample_idx,
            "test/total_timesteps_per_sample": timesteps,
            "test/final_error": error_val
        }, commit=(sample_idx < len(dataset_tst) - 1))


def main(config, model, dataset_trn, dataset_test):
    for epoch in range(config["num_epochs"]):
        train_epoch(config, model, dataset_trn, epoch)
        test_epoch(config, model, dataset_test, epoch)

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