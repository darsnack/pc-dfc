import wandb
import jax.numpy as jnp
import jax.random as jrng
import jax.tree as jtu
from flax import nnx
from tqdm.auto import tqdm

from projectlib.models import MLP, Activation
from projectlib.training import pc_step, current_pc_vjp, approx_pc_step
from projectlib.utils import noop

def is_converged(activations, tolerance):
    """Check if the model has converged based on the change in activations."""
    if len(activations[0]) < 2:
        return False
    else:
        dvs = jnp.array([jnp.linalg.norm(act[-1] - act[-2]) ** 2 for act in activations])
        return jnp.sqrt(jnp.sum(dvs)) < tolerance

def run(model, step_fn, x, max_timesteps = 5000, tolerance = 1e-5, disable_tqdm=False, **kwargs):
    """Run the model for a specified number of timesteps or until convergence."""
    errors = []
    # Store initial activations
    activations = [[a.value.copy()] for a in model.activations]
    t = 0
    # Loop with tqdm progress bar based on t and max_timesteps
    with tqdm(total=max_timesteps, desc="timesteps", disable=disable_tqdm) as bar:
        while (t < max_timesteps) and not is_converged(activations, tolerance):
            # Run single step of model dynamics
            error = step_fn(model, x, **kwargs)
            # Store activations at this timestep
            for acts, a in zip(activations, model.activations):
                acts.append(a.value.copy())
            errors.append(error)
            t += 1
            bar.update(1)

    errors = jnp.stack(errors)
    activations = [jnp.stack(acts) for acts in activations]

    return errors, activations, t

@nnx.jit
def get_model_params(model):
    """Extract all model parameters as a flat dictionary."""
    # Extract weights and biases from each layer
    filter = nnx.All(nnx.Param, nnx.Not(Activation))
    params = {k: v.value.copy()
              for k, v in nnx.variables(model, filter).flat_state()}

    return params

@nnx.jit
def compute_param_change_norm(old_params, new_params):
    """Compute the norm of parameter changes between two parameter sets."""
    # Compute squared differences for each parameter
    norm = jnp.sqrt(jtu.reduce(
        lambda x, y: x + y,
        jtu.map(lambda old, new: jnp.sum((new - old) ** 2), old_params, new_params)
    ))

    return norm

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

# All of below is training,
# TODO: add testing after
# TODO: do we need to store v_fp separately?
#       (see pseudo-code uses model.activations[-1] directly)

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

# Main training loop
print("\nTraining:")
with tqdm(range(num_epochs), desc="epochs", position=0, leave=True) as epoch_bar:
    for epoch in epoch_bar:
        epoch_nudge_counts = []
        epoch_timestep_counts = []
        epoch_errors = []
        with tqdm(enumerate(dataset_trn), desc="trn samples", position=1, leave=False, total=len(dataset_trn)) as sample_bar:
            for sample_idx, (x, y) in sample_bar:
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

                    # Update progress bar with current nudge info
                    sample_bar.set_postfix({
                        "nudge": nudge_count,
                        "timesteps": timesteps,
                        "error": f"{error_val:.6f}"
                    })

                    # Log each nudge step to wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/sample": sample_idx,
                        "train/nudge": nudge_count,
                        "train/timesteps_per_nudge": timesteps,
                        "train/error": error_val,
                        "train/weight_change_norm": float(param_change_norm)
                    }, commit=(error_val > nudge_eps))

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

        with tqdm(enumerate(dataset_tst), desc="tst samples", position=2, leave=False, total=len(dataset_tst)) as sample_bar:
            for sample_idx, (x, y) in sample_bar:
                # Run dynamics until convergence
                errors, activations, timesteps = run(model, approx_pc_step, x,
                                                     disable_tqdm=True,
                                                     vjps=vjps,
                                                     tau=tau,
                                                     eta=lr,
                                                     learn=False)
                # Update fixed point
                v_fp = activations[-1][-1]
                vjps = current_pc_vjp(model, x)

                # Record nudge statistics
                error_val = float(jnp.linalg.norm(v_fp - y))

                # Update progress bar with current nudge info
                sample_bar.set_postfix({
                    "timesteps": timesteps,
                    "error": f"{error_val:.6f}"
                })

                # # Log each nudge step to wandb
                # wandb.log({
                #     "epoch": epoch,
                #     "train/sample": sample_idx,
                #     "train/nudge": nudge_count,
                #     "train/timesteps_per_nudge": timesteps,
                #     "train/error": error_val,
                #     "train/weight_change_norm": float(param_change_norm)
                # }, commit=(error_val > nudge_eps))

                # Log sample-level metrics to wandb
                wandb.log({
                    "epoch": epoch,
                    "test/sample": sample_idx,
                    "test/total_timesteps_per_sample": timesteps,
                    "test/final_error": error_val
                }, commit=(sample_idx < len(dataset_tst) - 1))

        # Log epoch-level summary statistics
        wandb.log({
            "epoch": epoch,
            "avg_nudges_per_sample": sum(epoch_nudge_counts) / len(epoch_nudge_counts),
            "max_nudges": max(epoch_nudge_counts),
            "avg_timesteps_per_sample": sum(epoch_timestep_counts) / len(epoch_timestep_counts),
            "max_timesteps": max(epoch_timestep_counts),
            "avg_final_error": sum(epoch_errors) / len(epoch_errors)
        })

wandb.finish()
