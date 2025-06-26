import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from flax import nnx

from projectlib.models import MLP
from projectlib.training import pc_step, current_pc_vjp, approx_pc_step
from projectlib.utils import noop

# plotting helper function
def plot_dynamics(axs, errors, out_acts, hidden_acts, **kwargs):
    p1 = axs[0].plot(errors, **kwargs)
    axs[0].set_ylim(0, None)
    axs[0].set_title("Total energy")
    p2 = axs[1].plot(out_acts, **kwargs)
    axs[1].set_title("Output activations")
    p3 = axs[2].plot(hidden_acts, **kwargs)
    axs[2].set_title("Hidden activations")

    return (p1, p2, p3)

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

# initialize randomness
rngs = nnx.Rngs(0, params=1, train=2)

# build model
# model = MLP([20, 10, 8, 6, 4, 2], rngs=rngs)
model = MLP([20, 10, 8, 2], activation_fn=noop, rngs=rngs)
print(model)
y = model(jnp.ones((20,)))
print(y.shape)
print(model.activations[0])
print(model.activations[1])
print(model.activations_ff.value[0])
print(model.activations_ff.value[1])

# def dphi(x):
#     return 1 - nnx.tanh(x) ** 2
# gtrue = - model.layers[-1].kernel @ (model.activations[-1] - model.activations_ff[-1]) * dphi(model.activations[-2]) + model.activations[-2] - model.activations_ff[-2]

# print(model.activations[0] - gtrue)
# pc_step(model, jnp.ones((20,)))
# print(model.activations)

orig_activations = [act.value.copy() for act in model.activations]

x = jrng.normal(rngs(), (20,))
ntimesteps = 500
rclamp = jnp.ones((2,)) * 0.3
tau = 30
eta = 1e-3

errors, activations = run(model, pc_step, ntimesteps, x,
                          tau=tau, eta=eta, rclamp=rclamp)
print("Final error (training) = ", errors[-1])
fig, axs = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
plot_dynamics(axs[0, :], errors, activations[-1], activations[0])

for i, act in enumerate(orig_activations):
    model.activations[i].value = act
errors, activations = run(model, pc_step, ntimesteps, x,
                          tau=tau, inference=True)
print("Final error (inference) = ", errors[-1])
plot_dynamics(axs[1, :], errors, activations[-1], activations[0])

perturb = 0.1 * jrng.normal(rngs(), model.activations[-1].shape)
rclamp = model.activations[-1].value + perturb
w_orig = model.layers[1].kernel.value.copy()
fixed_pt_model = nnx.clone(model)
fixed_pt_activations = [act.value.copy() for act in model.activations]

errors, pcn_activations = run(model, pc_step, ntimesteps, x,
                              tau=tau, eta=eta, inference=False, rclamp=rclamp)
print("Final error (training w/ perturb, pcn) = ", errors[-1])
for i, act in enumerate(fixed_pt_activations):
    model.activations[i].value = act
errors, pcn_activations = run(model, pc_step, ntimesteps, x,
                              tau=tau, eta=eta, inference=True)
plot_dynamics(axs[2, :], errors, pcn_activations[-1], pcn_activations[0])

model = nnx.clone(fixed_pt_model)
vjps = current_pc_vjp(model, x)
errors, dfc_activations = run(model, approx_pc_step, ntimesteps, x,
                              vjps=vjps, tau=tau, eta=eta, inference=False, rclamp=rclamp)
print("Starting error (training w/ perturb, approx pcn) = ", errors[0])
print("Final error (training w/ perturb, approx pcn) = ", errors[-1])
for i, act in enumerate(fixed_pt_activations):
    model.activations[i].value = act
errors, pcn_activations = run(model, approx_pc_step, ntimesteps, x,
                              vjps=vjps, tau=tau, eta=eta, inference=True, rclamp=rclamp)
plot_dynamics(axs[2, :], errors, dfc_activations[-1], dfc_activations[0], linestyle="--")

true_line = mlines.Line2D([], [], color="black", linestyle="-", label="true")
approx_line = mlines.Line2D([], [], color="black", linestyle="--", label="approx")
plt.figlegend(handles=[true_line, approx_line],
              ncols=2,
              loc="outside lower center")
plt.show()

perturb_sizes = jnp.logspace(-1, 1, 20)
perturb = jrng.normal(rngs(), model.activations[-1].shape)
layer_figs, layer_axs = [], []
for i, act in enumerate(model.activations):
    fig, axs = plt.subplots(nrows=3, ncols=act.shape[-1],
                            figsize=(12, 10), layout="constrained")
    layer_figs.append(fig)
    layer_axs.append(axs)
err_fig, err_axs = plt.subplots(nrows=1, ncols=len(model.layers) + 1,
                                figsize=(12, 10), layout="constrained")
cmap = plt.get_cmap("copper")
colors = [cmap(i) for i in jnp.linspace(0.1, 1, len(perturb_sizes))]
layer_colors = [cmap(i) for i in jnp.linspace(0.1, 1, len(fixed_pt_activations))]
for p, perturb_size in enumerate(perturb_sizes):
    rclamp = fixed_pt_activations[-1] + perturb_size * perturb

    model = nnx.clone(fixed_pt_model)
    pcn_errors, pcn_activations = run(model, pc_step, ntimesteps, x,
                                      tau=tau, eta=eta, inference=False, rclamp=rclamp)
    print(f"Final error (training, pcn, {perturb_size=}) = ", pcn_errors[-1])
    for i, act in enumerate(fixed_pt_activations):
        model.activations[i].value = act
    pcn_errors, pcn_activations = run(model, pc_step, ntimesteps, x,
                                      tau=tau, eta=eta, inference=True)

    model = nnx.clone(fixed_pt_model)
    dfc_errors, dfc_activations = run(model, approx_pc_step, ntimesteps, x,
                                      vjps=vjps, tau=tau, eta=eta, inference=False, rclamp=rclamp)
    print(f"Final error (training, approx pcn, {perturb_size=}) = ", dfc_errors[-1])
    for i, act in enumerate(fixed_pt_activations):
        model.activations[i].value = act
    dfc_errors, dfc_activations = run(model, approx_pc_step, ntimesteps, x,
                                      vjps=vjps, tau=tau, eta=eta, inference=True, rclamp=rclamp)

    act_diffs = [pcn[-1] - dfc
                 for pcn, dfc in zip(pcn_activations, dfc_activations)]
    fx_pt_diffs = [pcn - fxp
                   for pcn, fxp in zip(pcn_activations, fixed_pt_activations)]

    for k, diffs in enumerate(act_diffs):
        for j in range(diffs.shape[-1]):
            layer_axs[k][0, j].plot(diffs[:, j], color=colors[p])
            layer_axs[k][1, j].scatter(perturb_size, diffs[-1, j], color=colors[p])
            layer_axs[k][2, j].scatter(perturb_size, fx_pt_diffs[k][-1, j], color=colors[p])
            layer_axs[k][1, j].set_xscale("log")
            layer_axs[k][2, j].set_xscale("log")
        err_axs[k].scatter(perturb_size, jnp.mean(jnp.abs(diffs[-1])), color=colors[p])
        err_axs[-1].scatter(perturb_size, jnp.mean(jnp.abs(diffs[-1])), color=layer_colors[k])
        err_axs[k].set_xscale("log")
    err_axs[-1].set_xscale("log")

plt.show()
