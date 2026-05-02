import jax
import jax.numpy as jnp
import jax.tree as jtr
from flax import nnx
from tqdm import tqdm

from projectlib.models import Activation

def pc_error(vs, vs_ff):
    # TODO: add loss here for PCN variant that can incorporate cross entropy
    min_samples = min(vs[0].shape[0], vs_ff[0].shape[0])
    total_energy = 0.5 * sum(jnp.sum((v[:min_samples] - v_ff[:min_samples]) ** 2, axis=-1)
                             for v, v_ff in zip(vs, vs_ff))

    return total_energy

@nnx.jit(static_argnames=("learn",))
def pc_step(model: nnx.Module, x, tau = 1, eta = 1, rclamp = None, learn = False):
    def _compute_pc_error(model, x, rclamp):
        _ = model(x, rclamp)

        return pc_error(model.activations, model.activations_ff)

    acts_only = Activation
    if not learn:
        # compute local gradients w.r.t. current activations
        # nnx.DiffState(...) means w.r.t. 0-th argument and only the leaves of the tree
        # that match the acts_only filter
        grad_fn = nnx.value_and_grad(_compute_pc_error, argnums=nnx.DiffState(0, acts_only))
    else:
        # compute local gradients w.r.t. to current activations + weights
        grad_fn = nnx.value_and_grad(_compute_pc_error)

    if x.ndim > 1:
        state_axes = nnx.StateAxes({Activation: 0, nnx.Intermediate: 0, ...: None})
        clamp_axes = None if rclamp is None else 0
        grad_fn = nnx.vmap(grad_fn, in_axes=(state_axes, 0, clamp_axes), out_axes=0)

    total_energy, grads = grad_fn(model, x, rclamp)
    # update only current activations
    acts_state, other_state = nnx.state(model, acts_only, nnx.Param)
    acts_grads, other_grads = nnx.split_state(grads, acts_only, nnx.Param)
    new_acts = jtr.map(lambda v, dv: v - dv / tau, acts_state, acts_grads)
    if learn:
        # TODO: should we update avg. across batches??
        other_state = jtr.map(lambda w, dw: w - eta * jnp.mean(dw, axis=0), other_state, other_grads)
    nnx.update(model, new_acts, other_state)

    return total_energy

@nnx.jit
def current_pc_vjp(model: nnx.Module, x):
    # get inputs for each layer
    vs = (x, *(v.value for v in model.activations[:-1]))
    # trick to call forward pass of split model = graph + state
    def call(v, graph, state):
        layer = nnx.merge(graph, state)
        return layer(v)
    # run forward direction for each layer, storing the VJPs w.r.t. input
    vjps = []
    for i, (v, layer) in enumerate(zip(vs, model.layers)):
        if i == 0:
            vjp = jax.vjp(lambda _v, _l=layer: call(_v, *nnx.split(_l)), v)[1]
        else:
            vjp = jax.vjp(lambda _v, _l=layer: call(model.activation_fn(_v), *nnx.split(_l)), v)[1]
        vjps.append(vjp)

    return vjps

@nnx.jit(static_argnames=("learn",))
def approx_pc_step(model: nnx.Module, x, vjps, k = 1, tau = 1, eta = 1, rclamp = None, learn = False):
    # evaluate bottom up activations
    _ = model(x, rclamp)
    total_energy = pc_error(model.activations, model.activations_ff)
    # compute error signal
    # note that r_final = identity(model.activations_ff[-1])
    # change this to reflect non-identity activation (e.g. softmax)
    final_error = model.activations[-1] - model.activations_ff[-1]
    # compute update for last layer (normal proportional control)
    # if rclamp is set, we do not update the last layer
    if rclamp is None:
        dvs = [(-model.activations[-1] + model.activations_ff[-1]) / tau]
    else:
        dvs = [0]
    dws = []
    # apply error propagation with VJPs starting from next to last layer
    # and working to the front of the network
    error = final_error
    for act, v_ff, vjp in zip(reversed(model.activations[:-1]),
                              reversed(model.activations_ff[:-1]),
                              reversed(vjps[1:])):
        # extract raw array from Activation variable so that JAX
        # functions like tanh can be called on it inside nnx.jit
        v = act.value
        # compute gradient w.r.t. weights
        _v = v if v.ndim == 2 else v[None, :]
        _e = error if error.ndim == 2 else error[None, :]
        dw = eta / _e.shape[0] * model.activation_fn(_v).T @ _e
        dws.append(dw)
        # propagate error backwards
        # (i.e. running prod_l J^l e^L)
        error = vjp(error)[0]
        # compute proportional control signal
        u = k * error
        # compute dynamics update for activations
        dv = (-v + v_ff + u) / tau
        dvs.append(dv)
    # compute gradient update for first layer
    _x = x if x.ndim == 2 else x[None, :]
    _e = error if error.ndim == 2 else error[None, :]
    dw = eta / _e.shape[0] * _x.T @ _e
    dws.append(dw)

    # apply dynamics
    for v, dv in zip(model.activations, reversed(dvs)):
        v.value = v.value + dv
    if learn:
        for layer, dw in zip(model.layers, reversed(dws)):
            layer.kernel.value = layer.kernel.value + dw

    return jnp.mean(total_energy)


def run(model, step_fn, x,
        min_timesteps = 5,
        max_timesteps = 5000,
        tolerance = 1e-5,
        tolerance_style = "norm",
        disable_tqdm = False,
        return_act_history = False,
        nmodels = 1,
        **kwargs):
    """Run the model for a specified number of timesteps or until convergence."""
    errors = []
    # Store initial activations
    activations = [[a.value.copy()] for a in model.activations]
    t = 0
    # Loop with tqdm progress bar based on t and max_timesteps
    with tqdm(total=max_timesteps, desc="timesteps", disable=disable_tqdm) as bar:
        while (t < min_timesteps) or ((t < max_timesteps)
                                      and not is_converged(activations, tolerance, tolerance_style, nmodels)):
            # Run single step of model dynamics
            error = step_fn(model, x, **kwargs)
            # Store activations at this timestep
            for acts, a in zip(activations, model.activations):
                acts.append(a.value.copy())
            errors.append(error)
            t += 1
            bar.update(1)

    errors = jnp.stack(errors)
    if return_act_history:
        activations = [jnp.stack(acts) for acts in activations]
    else:
        activations = [acts[-1] for acts in activations]

    return errors, activations, t


def is_converged(activations, tolerance, tolerance_style, nmodels=1):
    """Check if the model has converged based on the change in activations."""
    if len(activations[0]) < 2:
        return False
    else:
        # dvs is [L, ...] where trailing dims are (nmodels, B) or (B,)
        dvs = jnp.array([jnp.sum((act[-1] - act[-2]) ** 2, axis=-1) for act in activations])
        dv_total = jnp.sum(dvs, axis=0)
        if tolerance_style == "norm":
            dist = jnp.sqrt(dv_total)
        else:
            n_neurons = sum([act.shape[-1] for act in activations])
            dist = dv_total / n_neurons

        if nmodels > 1:
            return jnp.max(jnp.min(dist, axis=-1)) < tolerance
        else:
            return jnp.min(dist) < tolerance
