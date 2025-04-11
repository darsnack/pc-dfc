import jax
import jax.random as jrng
from flax import nnx
from typing import Sequence, Callable

from projectlib.utils import noop

class MLP(nnx.Module):
    def __init__(self, features: Sequence[int], *, rngs,
              activation_fn: Callable = nnx.tanh, stop_gradients: bool = True):
        rng_act = jrng.split(rngs.params(), len(features) - 1)
        self.layers = [nnx.Linear(fin, fout, rngs=rngs)
                       for fin, fout in zip(features[:-1], features[1:])]
        self.activations = [nnx.Param(jrng.normal(k, (fout,)))
                            for k, fout in zip(rng_act, features[1:])]
        # self.stopme_maybe = jax.lax.stop_gradient if stop_gradients else noop
        self.activation_fn = activation_fn

    def __call__(self, x):
        rs = [x, *(self.activation_fn(v.value) for v in self.activations[:-1])]
        vs = [layer(r) for layer, r in zip(self.layers, rs)]

        self.sow(nnx.Intermediate, "activations_ff", vs,
                 init_fn=(lambda: []),
                 reduce_fn=(lambda _, new: new))

        return vs[-1]
