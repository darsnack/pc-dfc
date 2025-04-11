import jax
import jax.tree as jtu
import optax
# import torch
from flax import nnx
# from torchvision.transforms import v2 as transforms
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt

from projectlib.models import MLP
# from projectlib.data import (load_dataset,
#                              build_dataloader,
#                              OneHot,
#                              JaxImageOrder)

# initialize randomness
rngs = nnx.Rngs(0, params=1, train=2)

# load dataset
# img_tfms = transforms.Compose([
#     transforms.ToImage(),
#     transforms.ToDtype(torch.float32, scale=True),
#     JaxImageOrder()
# ])
# lbl_tfms = OneHot(10)
# data = load_dataset("mnist", (img_tfms, lbl_tfms))
# train_loader = build_dataloader(data["train"], batch_size=128)
# test_loader = build_dataloader(data["test"], batch_size=128)

# build model
model = MLP([20, 10, 2], rngs=rngs)
print(model)
y = model(jnp.ones((20,)))
print(y.shape)
print(model.activations[0])
print(model.activations[1])
print(model.activations_ff.value[0])
print(model.activations_ff.value[1])

def compute_pc_error(model, x, rclamp = None):
    _ = model(x)
    vs_ff = model.activations_ff
    if rclamp is None:
        vs = model.activations
    else:
        vs = [*(model.activations[:-1]), rclamp]
    E = 0.5 * sum([jnp.sum((v - v_ff) ** 2) for v, v_ff in zip(vs, vs_ff)])

    return E

print(compute_pc_error(model, jnp.ones((20,))))
g = nnx.grad(compute_pc_error)(model, jnp.ones((20,)))
print(g)

def dphi(x):
    return 1 - nnx.tanh(x) ** 2
gtrue = - model.layers[-1].kernel @ (model.activations[-1] - model.activations_ff[-1]) * dphi(model.activations[-2]) + model.activations[-2] - model.activations_ff[-2]
print(gtrue)

def pc_step(model, x, tau = 1, rclamp = None):
    E, grad = nnx.value_and_grad(compute_pc_error)(model, x, rclamp=rclamp)
    for i, (a, g) in enumerate(zip(model.activations, grad.activations.values())):
        model.activations[i].value = a.value - g.value / tau

    return E

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
    print(error)
    activations[0].append(model.activations[0].value.copy())
    activations[1].append(model.activations[1].value.copy())
    errors.append(error)

fig, axs = plt.subplots(1, 3)
axs[0].plot(errors)
axs[1].plot(activations[1])
axs[2].plot(activations[0])
plt.show()
