import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from flax import nnx
from numpy import append

from projectlib.models import MLP
from projectlib.training import pc_step, current_pc_vjp, approx_pc_step

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
model = MLP([20, 10, 2], rngs=rngs)
print(model)
y = model(jnp.ones((20,)))
print(y.shape)
print(model.activations[0])
print(model.activations[1])
print(model.activations_ff.value[0])
print(model.activations_ff.value[1])

def dphi(x):
    return 1 - nnx.tanh(x) ** 2
gtrue = - model.layers[-1].kernel @ (model.activations[-1] - model.activations_ff[-1]) * dphi(model.activations[-2]) + model.activations[-2] - model.activations_ff[-2]

print(model.activations[0] - gtrue)
pc_step(model, jnp.ones((20,)))
print(model.activations)

orig_activations = [model.activations[0].value.copy(),
                    model.activations[1].value.copy()]

x = jrng.normal(rngs(), (20,))
ntimesteps = 500
rclamp = jnp.ones((2,)) * 0.3

errors, activations = run(model, pc_step, ntimesteps, x,
                          tau=10, eta=1e-3, rclamp=rclamp)
print("Final error (training) = ", errors[-1])
fig, axs = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
plot_dynamics(axs[0, :], errors, activations[1], activations[0])

model.activations[0].value = orig_activations[0]
model.activations[1].value = orig_activations[1]
errors, activations = run(model, pc_step, ntimesteps, x,
                          tau=10, inference=True)
print("Final error (inference) = ", errors[-1])
plot_dynamics(axs[1, :], errors, activations[1], activations[0])

perturb = 1 * jrng.normal(rngs(), model.activations[-1].shape)
rclamp = model.activations[-1].value + perturb
fixed_pt_activations = [model.activations[0].value.copy(),
                        model.activations[1].value.copy()]

errors, pcn_activations = run(model, pc_step, ntimesteps, x,
                              tau=10, inference=True, rclamp=rclamp)
print("Final error (inference, true) = ", errors[-1])
plot_dynamics(axs[2, :], errors, pcn_activations[1], pcn_activations[0])

vjps = current_pc_vjp(model, x)
model.activations[0].value = fixed_pt_activations[0]
model.activations[1].value = fixed_pt_activations[1]
errors, dfc_activations = run(model, approx_pc_step, ntimesteps, x,
                              vjps=vjps, tau=10, rclamp=rclamp)
print("Final error (inference, approx) = ", errors[-1])
plot_dynamics(axs[2, :], errors, dfc_activations[1], dfc_activations[0], linestyle="--")

true_line = mlines.Line2D([], [], color="black", linestyle="-", label="true")
approx_line = mlines.Line2D([], [], color="black", linestyle="--", label="approx")
plt.figlegend(handles=[true_line, approx_line],
              ncols=2,
              loc="outside lower center")
plt.show()

perturb_sizes = jnp.linspace(0.05, 5, 10)
perturb = jrng.normal(rngs(), model.activations[-1].shape)
fig, axs = plt.subplots(nrows=4, ncols=3)
cmap = plt.get_cmap("Blues")
colors = [cmap(i) for i in jnp.linspace(0, 1, len(perturb_sizes))]
for i, perturb_size in enumerate(perturb_sizes):
    rclamp = fixed_pt_activations[-1] + perturb_size * perturb

    model.activations[0].value = fixed_pt_activations[0]
    model.activations[1].value = fixed_pt_activations[1]
    pcn_errors, pcn_activations = run(model, pc_step, ntimesteps, x,
                                      tau=10, inference=True, rclamp=rclamp)
    print(f"Final error (inference, true, {perturb_size=}) = ", pcn_errors[-1])

    model.activations[0].value = fixed_pt_activations[0]
    model.activations[1].value = fixed_pt_activations[1]
    dfc_errors, dfc_activations = run(model, approx_pc_step, ntimesteps, x,
                                      vjps=vjps, tau=10, rclamp=rclamp)
    print(f"Final error (inference, approx, {perturb_size=}) = ", dfc_errors[-1])

    act_diffs = [pcn - dfc for pcn, dfc in zip(pcn_activations, dfc_activations)]

    for j in range(act_diffs[0].shape[-1]):
        axs.flat[j].plot(act_diffs[0][:, j], color=colors[i])

plt.show()
