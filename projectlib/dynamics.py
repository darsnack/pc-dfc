import jax
import jax.numpy as jnp
import jax.tree as jtr
from flax import nnx
from tqdm import tqdm

from projectlib.models import Activation

def pc_error(vs, vs_ff):
    # TODO: add loss here for PCN variant that can incorporate cross entropy
    return 0.5 * sum(jnp.sum((v - v_ff) ** 2)
                     for v, v_ff in zip(vs, vs_ff))

@nnx.jit(static_argnames=("learn", "batched", "loss_fn"))
def pc_step(model: nnx.Module, opt: nnx.Optimizer, x, tau = 1, rclamp = None,
            learn = False, loss_fn = None, batched = False):
    if loss_fn is None:
        def _compute_pc_error(model, x, rclamp):
            pred = model(x, rclamp)

            return pc_error(model.activations, model.activations_ff), pred
    else:
        def _compute_pc_error(model, x, rclamp):
            pred = model(x, rclamp)
            free_energy = pc_error(model.activations[:-1], model.activations_ff[:-1])
            # when rclamp != None, model.activations[-1] == rclamp
            final_error = loss_fn(model.activations_ff[-1], model.activations[-1])

            return free_energy + final_error, pred

    acts_only = Activation
    if not learn:
        # compute local gradients w.r.t. current activations
        # nnx.DiffState(...) means w.r.t. 0-th argument and only the leaves of the tree
        # that match the acts_only filter
        grad_fn = nnx.value_and_grad(_compute_pc_error,
                                     argnums=nnx.DiffState(0, acts_only),
                                     has_aux=True)
    else:
        # compute local gradients w.r.t. to current activations + weights
        grad_fn = nnx.value_and_grad(_compute_pc_error,
                                     argnums=nnx.DiffState(0, (acts_only, nnx.Param)),
                                     has_aux=True)

    if batched:
        state_axes = nnx.StateAxes({Activation: 0, nnx.Intermediate: 0, ...: None})
        clamp_axes = None if rclamp is None else 0
        grad_fn = nnx.vmap(grad_fn, in_axes=(state_axes, 0, clamp_axes), out_axes=0)

    (total_energy, pred), grads = grad_fn(model, x, rclamp)
    # update only current activations
    acts_state, other_state = nnx.state(model, acts_only, nnx.Param)
    acts_grads, other_grads = nnx.split_state(grads, acts_only, nnx.Param)
    new_acts = jtr.map(lambda v, dv: v - dv / tau, acts_state, acts_grads)
    if learn:
        # TODO: should we update avg. across batches??
        other_grads = jtr.map(lambda dw: jnp.mean(dw, axis=0), other_grads)
        opt.update(other_grads)
        # other_state = jtr.map(lambda w, dw: w - eta * jnp.mean(dw, axis=0), other_state, other_grads)
    nnx.update(model, new_acts)

    return total_energy, pred

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

@nnx.jit
def current_param_vjp(model: nnx.Module, x):
    layer_inputs = (x, *(model.activation_fn(v.value)
                         for v in model.activations[:-1]))
    param_vjps = []
    for layer, layer_input in zip(model.layers, layer_inputs):
        graph, params, other_state = nnx.split(layer, nnx.Param, ...)
        # trick to call forward pass of split model = graph + state
        def call(_params, _graph=graph, _other_state=other_state,
                 _layer_input=layer_input):
            layer_with_params = nnx.merge(_graph, _params, _other_state)
            return layer_with_params(_layer_input)
        # run forward direction for each layer, storing the VJPs w.r.t. params
        param_vjps.append(jax.vjp(call, params)[1])

    return param_vjps

@nnx.jit(static_argnames=("learn", "batched"))
def approx_pc_step(model: nnx.Module, opt: nnx.Optimizer, x, vjps,
                   k = 1, tau = 1, rclamp = None, learn = False,
                   batched = False):
    # evaluate bottom up activations
    pred = model(x, rclamp)
    if batched:
        total_energy = jax.vmap(pc_error)(
            tuple(v.value for v in model.activations),
            tuple(model.activations_ff))
    else:
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
    param_vjps = current_param_vjp(model, x) if learn else None
    # errors are collected in forward layer order below.
    layer_errors = [None] * len(model.layers)
    # apply error propagation with VJPs starting from next to last layer
    # and working to the front of the network
    error = final_error
    for layer_idx, (act, v_ff, vjp) in enumerate(zip(reversed(model.activations[:-1]),
                                                     reversed(model.activations_ff[:-1]),
                                                     reversed(vjps[1:]))):
        # extract raw array from Activation variable so that JAX
        # functions like tanh can be called on it inside nnx.jit
        v = act.value
        layer_errors[len(model.layers) - 1 - layer_idx] = error
        # propagate error backwards
        # (i.e. running prod_l J^l e^L)
        error = vjp(error)[0]
        # compute proportional control signal
        u = k * error
        # compute dynamics update for activations
        dv = (-v + v_ff + u) / tau
        dvs.append(dv)
    # error at the output of the first layer.
    layer_errors[0] = error

    # take parameter vjps BEFORE updating activation states
    if learn:
        grads = nnx.map_state(
            lambda _, state: state.replace(jnp.zeros_like(state.value)),
            nnx.state(model, nnx.Param))
        for layer_idx, (param_vjp, layer_error) in enumerate(zip(param_vjps, layer_errors)):
            batch_size = x.shape[0] if batched else 1
            # the vjp already contracts over every output axis, including the batch axis.
            # below is equivalent to:
            #   tree_map(lambda g: -g / batch_size, param_vjp(layer_error)[0])
            grads["layers"][layer_idx] = param_vjp(-layer_error / batch_size)[0]

    # apply dynamics
    for v, dv in zip(model.activations, reversed(dvs)):
        v.value = v.value + dv
    if learn:
        opt.update(grads)

    return jnp.mean(total_energy), pred

def run(model, step_fn, x,
        min_timesteps = 5,
        max_timesteps = 5000,
        tolerance = 1e-5,
        tolerance_style = "norm",
        disable_tqdm = False,
        return_act_history = False,
        nmodels = 1,
        batched = False,
        **kwargs):
    """Run the model for a specified number of timesteps or until convergence."""
    errors = []
    # Store initial activations
    activations = [[a.value.copy()] for a in model.activations]
    t = 0
    # Loop with tqdm progress bar based on t and max_timesteps
    with tqdm(total=max_timesteps, desc="timesteps", disable=disable_tqdm) as bar:
        while (t < min_timesteps) or ((t < max_timesteps)
                                      and not is_converged(activations, tolerance,
                                                           tolerance_style, nmodels,
                                                           batched)):
            # Run single step of model dynamics
            error, pred = step_fn(model, x, batched=batched, **kwargs)
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

    return pred, errors, activations, t


def is_converged(activations, tolerance, tolerance_style, nmodels=1,
                 batched=False):
    """Check if the model has converged based on the change in activations."""
    if len(activations[0]) < 2:
        return False
    else:
        # Preserve model and batch axes; reduce every activation/neuron axis.
        leading_axes = int(nmodels > 1) + int(batched)
        dvs = jnp.array([
            jnp.sum((act[-1] - act[-2]) ** 2,
                    axis=tuple(range(leading_axes, act[-1].ndim)))
            for act in activations])
        dv_total = jnp.sum(dvs, axis=0)
        if tolerance_style == "norm":
            dist = jnp.sqrt(dv_total)
        else:
            n_neurons = sum(
                int(jnp.prod(jnp.array(act[-1].shape[leading_axes:])))
                for act in activations)
            dist = dv_total / n_neurons

        if nmodels > 1:
            if batched:
                return jnp.max(jnp.min(dist, axis=-1)) < tolerance
            return jnp.max(dist) < tolerance
        else:
            if batched:
                return jnp.min(dist) < tolerance
            return dist < tolerance
