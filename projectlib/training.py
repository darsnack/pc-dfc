import jax.numpy as jnp
import jax.tree as jtr
from flax import nnx

def pc_error(vs, vs_ff):
    total_energy = 0.5 * sum(jnp.sum((v - v_ff) ** 2)
                             for v, v_ff in zip(vs, vs_ff))

    return total_energy

def pc_step(model: nnx.Module, x, tau = 1, eta = 1, rclamp = None, inference = False):
    def _compute_pc_error(model, x, rclamp):
        _ = model(x, rclamp)

        return pc_error(model.activations, model.activations_ff)

    acts_only = nnx.PathContains("activations")
    if inference:
        # compute local gradients w.r.t. current activations
        # nnx.DiffState(...) means w.r.t. 0-th argument and only the leaves of the tree
        # that match the acts_only filter
        grad_fn = nnx.value_and_grad(_compute_pc_error, argnums=nnx.DiffState(0, acts_only))
    else:
        # compute local gradients w.r.t. to current activations + weights
        grad_fn = nnx.value_and_grad(_compute_pc_error)

    total_energy, grads = grad_fn(model, x, rclamp)
    # update only current activations
    # print(grads)
    _, acts_state, other_state = nnx.split(model, acts_only, ...)
    # acts_grads, other_grads = nnx.state(grads, acts_only, ...)
    # print(acts_state, acts_grads)
    new_acts = jtr.map(lambda v, dv: v - dv / tau, acts_state["activations"], grads["activations"])
    if not inference:
        other_state = jtr.map(lambda w, dw: w - eta * dw, other_state["layers"], grads["layers"])
        other_state = {"layers": other_state}
    nnx.update(model, {"activations": new_acts}, other_state)

    return total_energy
