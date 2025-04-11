import jax.numpy as jnp
from flax import nnx

def pc_error(vs, vs_ff):
    total_energy = 0.5 * sum(jnp.sum((v - v_ff) ** 2)
                             for v, v_ff in zip(vs, vs_ff))

    return total_energy

def pc_step(model, x, tau = 1, rclamp = None):
    def _compute_pc_error(model, x, rclamp):
        _ = model(x, rclamp)

        return pc_error(model.activations, model.activations_ff)

    total_energy, grads = nnx.value_and_grad(_compute_pc_error)(model, x, rclamp)
    for i, (a, g) in enumerate(zip(model.activations, grads.activations.values())):
        model.activations[i].value = a.value - g.value / tau

    return total_energy
