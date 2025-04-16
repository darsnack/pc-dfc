import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from flax import nnx

from projectlib.models import MLP
from projectlib.training import pc_step, current_pc_vjp, approx_pc_step

# plotting helper function
def plot_dynamics(axs, errors, out_acts, hidden_acts, **kwargs):
    p1 = axs[0].plot(errors, **kwargs)
    axs[0].set_ylim(0, None)
    axs[0].set_title("Total energy")
    p2 = axs[1].plot(activations[1], **kwargs)
    axs[1].set_title("Output activations")
    p3 = axs[2].plot(activations[0], **kwargs)
    axs[2].set_title("Hidden activations")

    return (p1, p2, p3)

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
errors = []
activations = [[model.activations[0].value.copy()],
               [model.activations[1].value.copy()]]
for t in range(ntimesteps):
    error = pc_step(model, x, tau=10, eta=1e-3, rclamp=rclamp)
    activations[0].append(model.activations[0].value.copy())
    activations[1].append(model.activations[1].value.copy())
    errors.append(error)

print("Final error (training) = ", errors[-1])

fig, axs = plt.subplots(3, 3, figsize=(8, 8), layout="constrained")
plot_dynamics(axs[0, :], errors, activations[1], activations[0])

errors = []
model.activations[0].value = orig_activations[0]
model.activations[1].value = orig_activations[1]
activations = [[model.activations[0].value.copy()],
               [model.activations[1].value.copy()]]
for t in range(ntimesteps):
    error = pc_step(model, x, tau=10, inference=True)
    activations[0].append(model.activations[0].value.copy())
    activations[1].append(model.activations[1].value.copy())
    errors.append(error)

print("Final error (inference) = ", errors[-1])
plot_dynamics(axs[1, :], errors, activations[1], activations[0])

perturb = 1e-1 * jrng.normal(rngs(), model.activations[-1].shape)
rclamp = model.activations[-1].value + perturb
orig_activations = [model.activations[0].value.copy(),
                    model.activations[1].value.copy()]

errors = []
activations = [[model.activations[0].value.copy()],
               [model.activations[1].value.copy()]]
for t in range(ntimesteps):
    error = pc_step(model, x, tau=10, inference=True, rclamp=rclamp)
    activations[0].append(model.activations[0].value.copy())
    activations[1].append(model.activations[1].value.copy())
    errors.append(error)

print("Final error (inference, true) = ", errors[-1])
plot_dynamics(axs[2, :], errors, activations[1], activations[0])

vjps = current_pc_vjp(model, x)
errors = []
model.activations[0].value = orig_activations[0]
model.activations[1].value = orig_activations[1]
activations = [[model.activations[0].value.copy()],
               [model.activations[1].value.copy()]]
for t in range(ntimesteps):
    error = approx_pc_step(model, x, vjps, tau=10, rclamp=rclamp)
    activations[0].append(model.activations[0].value.copy())
    activations[1].append(model.activations[1].value.copy())
    errors.append(error)

print("Final error (inference, approx) = ", errors[-1])
plot_dynamics(axs[2, :], errors, activations[1], activations[0], linestyle="--")

true_line = mlines.Line2D([], [], color="black", linestyle="-", label="true")
approx_line = mlines.Line2D([], [], color="black", linestyle="--", label="approx")
plt.figlegend(handles=[true_line, approx_line],
              ncols=2,
              loc="outside lower center")
plt.show()
