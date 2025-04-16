import jax.numpy as jnp
import jax.random as jrng
import matplotlib.pyplot as plt
from flax import nnx

from projectlib.models import MLP
from projectlib.training import pc_step

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
ntimesteps = 1000
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

fig, axs = plt.subplots(2, 3, figsize=(12, 8))
axs[0, 0].plot(errors)
axs[0, 0].set_ylim(0, None)
axs[0, 0].set_title("Total energy")
axs[0, 1].plot(activations[1])
axs[0, 1].set_title("Output activations")
axs[0, 2].plot(activations[0])
axs[0, 2].set_title("Hidden activations")

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

axs[1, 0].plot(errors)
axs[1, 0].set_ylim(0, None)
axs[1, 0].set_title("Total energy")
axs[1, 1].plot(activations[1])
axs[1, 1].set_title("Output activations")
axs[1, 2].plot(activations[0])
axs[1, 2].set_title("Hidden activations")

fig.tight_layout()
plt.show()
