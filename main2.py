import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from flax import nnx

from projectlib.models import MLP
from projectlib.training import pc_step, current_pc_vjp, approx_pc_step
from projectlib.utils import noop

rngs = nnx.Rngs(0, params=1, train=2)


def run(model, step_fn, ntimesteps, *args, **kwargs):
    errors = []
    activations = [[a.value.copy()] for a in model.activations]
    for t in range(ntimesteps):
        error = step_fn(model, *args, **kwargs)
        for acts, a in zip(activations, model.activations):
            acts.append(a.value.copy())
        errors.append(error)

    errors = jnp.stack(errors)
    activations = [jnp.stack(acts) for acts in activations]

    return errors, activations


model = MLP([20, 10, 8, 2], activation_fn=noop, rngs=rngs)


num_epochs = 100
tau = 30
eta = 1e-3

errors, activations = run(model, pc_step, convergence_eps, dataset_tbd[0][0], 
                          tau=tau, eta=eta, inference=True)

v_fp = activations[-1][-1]

for n in range(num_epochs):

    for (x, y) in dataset_tbd:

        while jnp.linalg.norm(v_fp - y) > eps:

            rclamp = v_fp + (y - v_fp) / c

            errors, activations = run(model, approx_pc_step, convergence_eps, x, 
                          tau=tau, eta=eta, inference=False, rclamp=rclamp)



            


