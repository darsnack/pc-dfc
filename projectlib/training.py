import jax
import jax.numpy as jnp
import jax.tree as jtr
from flax import nnx

from projectlib.models import Activation

def pc_error(vs, vs_ff):
    total_energy = 0.5 * sum(jnp.sum((v - v_ff) ** 2)
                             for v, v_ff in zip(vs, vs_ff))

    return total_energy

def pc_step(model: nnx.Module, x, tau = 1, eta = 1, rclamp = None, inference = False):
    def _compute_pc_error(model, x, rclamp):
        _ = model(x, rclamp)

        return pc_error(model.activations, model.activations_ff)

    acts_only = Activation
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
    acts_state, other_state = nnx.state(model, acts_only, nnx.Param)
    acts_grads, other_grads = nnx.split_state(grads, acts_only, nnx.Param)
    new_acts = jtr.map(lambda v, dv: v - dv / tau, acts_state, acts_grads)
    if not inference:
        other_state = jtr.map(lambda w, dw: w - eta * dw, other_state, other_grads)
    nnx.update(model, new_acts, other_state)

    return total_energy

def current_pc_vjp(model: nnx.Module, x):
    # get inputs for each layer
    vs = (x, *(v.value for v in model.activations[:-1]))
    # trick to call forward pass of split model = graph + state
    def call(v, graph, state):
        layer = nnx.merge(graph, state)
        return layer(v)
    # run forward direction for each layer, storing the VJPs w.r.t. input
    vjps = [jax.vjp(lambda _v: call(_v, *nnx.split(layer)), v)[1]
            for v, layer in zip(vs, model.layers)]

    return vjps

def approx_pc_step(model: nnx.Module, x, vjps, k = 1, tau = 1, eta = 1, rclamp = None, inference = True):
    # evaluate bottom up activations
    _ = model(x, rclamp)
    total_energy = pc_error(model.activations, model.activations_ff)
    # compute error signal
    # note that r_final = identity(model.activations_ff[-1])
    final_error = rclamp - model.activations_ff[-1]
    # compute update for last layer (normal proportional control)
    dvs = [k * (rclamp - model.activations[-1].value) / tau]
    dws = []
    # apply error propagation with VJPs starting from next to last layer
    # and working to the front of the network
    error = final_error
    for v, v_ff, vjp in zip(reversed(model.activations[:-1]),
                            reversed(model.activations_ff[:-1]),
                            reversed(vjps[1:])):
        # propagate error backwards
        # (i.e. running prod_l J^l e^L)
        error = vjp(error)[0]
        # compute proportional control signal
        u = k * error
        # compute dynamics update for activations
        dv = (-v + v_ff + u) / tau
        dvs.append(dv)

        dw = eta * error @ model.activation_fn(v).T
        dws.append(dw)

    # apply dynamics
    for v, dv in zip(model.activations, reversed(dvs)):
        v.value = v.value + dv

    if not inference:
        for w, dw in zip(model.layers, reversed(dws)):
            w.value = w.value + dw

    return total_energy
