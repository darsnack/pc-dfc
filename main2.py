import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from flax import nnx

from projectlib.models import MLP
from projectlib.training import pc_step, current_pc_vjp, approx_pc_step
from projectlib.utils import noop

def is_converged(activations, tolerance):
    """Check if the model has converged based on the change in activations."""
    dvs = [jnp.linalg.norm(act[-1] - act[-2]) ** 2 for act in activations]

    return jnp.sqrt(jnp.sum(dvs)) < tolerance

def run(model, step_fn, max_timesteps = 1000, tolerance = 1e-5, *args, **kwargs):
    """Run the model for a specified number of timesteps or until convergence."""
    errors = []
    # Store initial activations
    activations = [[a.value.copy()] for a in model.activations]
    t = 0
    while (t < max_timesteps) or not is_converged(activations, tolerance):
        # Run single step of model dynamics
        error = step_fn(model, *args, **kwargs)
        # Store activations at this timestep
        for acts, a in zip(activations, model.activations):
            acts.append(a.value.copy())
        errors.append(error)
        t += 1

    if t == max_timesteps:
        print(f"WARNING: maximum number of timesteps ({max_timesteps}) reached without convergence.")

    errors = jnp.stack(errors)
    activations = [jnp.stack(acts) for acts in activations]

    return errors, activations

# Initialize random number generator
rngs = nnx.Rngs(0, params=1, train=2)

# Build model
model = MLP([20, 10, 8, 2], activation_fn=noop, rngs=rngs)

# Training parameters
num_epochs = 100
tau = 30
lr = 1e-3
nudge_eps = 1e-5
nudge_factor = 1


# All of below is training,
# TODO: add testing after
# TODO: define dataset_tbd
# TODO: do we need to store v_fp separately?
#       (see pseudo-code uses model.activations[-1] directly)

# Converge to fixed point initially on first example
errors, activations = run(model, pc_step, dataset_tbd[0][0],
                          tau=tau, eta=lr, learn=False)
# Store fixed point of last layer
v_fp = activations[-1][-1]

# Main training loop
for n in range(num_epochs):
    for (x, y) in dataset_tbd:
        # Keep nudging toward current sample until convergence
        while jnp.linalg.norm(v_fp - y) > nudge_eps:
            # Compute nudge in direction of target/label
            rclamp = v_fp + (y - v_fp) / nudge_factor
            # Run dynamics until convergence
            errors, activations = run(model, approx_pc_step, x,
                                      tau=tau, eta=eta, learn=True, rclamp=rclamp)
            # Update fixed point
            v_fp = activations[-1][-1]
