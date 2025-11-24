import jax.random as jrng
from flax import nnx
from typing import Sequence, Callable

class Activation(nnx.Param):
    pass

class MLP(nnx.Module):
    def __init__(self, features: Sequence[int], *, rngs, activation_fn: Callable = nnx.tanh):
        self.layers = [nnx.Linear(fin, fout, use_bias=False, rngs=rngs)
                       for fin, fout in zip(features[:-1], features[1:])]
        # initialize the neuron activations (pre-nonlinearity)
        self.activations = [Activation(jrng.normal(rngs.params(), (fout,)))
                            for fout in features[1:]]
        self.activation_fn = activation_fn

    def __call__(self, x, rclamp = None):
        # clamp final layer activation to passed value
        if rclamp is not None:
            self.activations[-1].value = rclamp
        # create input to each layer as current activations
        # through non-linearity (except input)
        rs = [x, *(self.activation_fn(v.value) for v in self.activations[:-1])]
        # evaluate each layer to get feedforward (bottom up) activations
        vs = [layer(r) for layer, r in zip(self.layers, rs)]
        # store bottom up activations in intermediate variable for later
        self.sow(nnx.Intermediate, "activations_ff", vs,
                 init_fn=(lambda: []),
                 reduce_fn=(lambda _, new: new))

        return vs[-1] # activated with identity function
    
    def set_activations(self, activations):
        for i, act in enumerate(activations):
            self.activations[i].value = act
