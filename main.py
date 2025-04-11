import jax.numpy as jnp
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
print(gtrue)

print(model.activations[0] - gtrue)
pc_step(model, jnp.ones((20,)))
print(model.activations)

x = jnp.ones((20,))
rclamp = jnp.ones((2,)) * 0.3
errors = []
activations = [[model.activations[0].value.copy()],
               [model.activations[1].value.copy()]]
for t in range(100):
    error = pc_step(model, x, tau=10, rclamp=rclamp)
    activations[0].append(model.activations[0].value.copy())
    activations[1].append(model.activations[1].value.copy())
    errors.append(error)

print("Final error = ", errors[-1])

fig, axs = plt.subplots(1, 3)
axs[0].plot(errors)
axs[0].set_ylim(0, None)
axs[1].plot(activations[1])
axs[2].plot(activations[0])
plt.show()
