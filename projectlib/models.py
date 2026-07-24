import jax.numpy as jnp
import jax.random as jrng
from flax import nnx
from flax.nnx.nn.initializers import variance_scaling, DtypeLikeInexact
from typing import Sequence, Callable

class Activation(nnx.Variable):
    pass

def lecun_norm_w_scale(scale: float = 1.0,
                       in_axis: int | Sequence[int] = -2,
                       out_axis: int | Sequence[int] = -1,
                       batch_axis: Sequence[int] = (),
                       dtype: DtypeLikeInexact = jnp.float_):
    return variance_scaling(scale, "fan_in", "truncated_normal", in_axis=in_axis,
                            out_axis=out_axis, batch_axis=batch_axis, dtype=dtype)

class MLP(nnx.Module):
    def __init__(self, features: Sequence[int], *, rngs,
                 kernel_init_scale: float = 1.0,
                 activation_fn: Callable = nnx.tanh,
                 use_bias = True):
        self.layers = [nnx.Linear(fin, fout,
                                  use_bias=use_bias,
                                  kernel_init=lecun_norm_w_scale(scale=kernel_init_scale),
                                  rngs=rngs)
                       for fin, fout in zip(features[:-1], features[1:])]
        self.use_bias = use_bias
        # initialize the neuron activations (pre-nonlinearity)
        self.activations = [Activation(jrng.normal(rngs.activations(), (fout,)))
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

    def get_activations(self):
        return [act.value for act in self.activations]

    def set_activations(self, activations):
        for i, act in enumerate(activations):
            self.activations[i].value = act

class _ConvReadout(nnx.Module):
    """Flatten channels-last convolutional features before a linear readout."""
    def __init__(self, input_features: int, output_features: int, *, rngs,
                 kernel_init, use_bias):
        self.linear = nnx.Linear(input_features, output_features,
                                 use_bias=use_bias,
                                 kernel_init=kernel_init,
                                 rngs=rngs)

    def __call__(self, x):
        return self.linear(x.reshape((*x.shape[:-3], -1)))


class _ConvPool(nnx.Module):
    """A convolutional stage followed by stride-two spatial max pooling."""
    def __init__(self, input_features: int, output_features: int, *, rngs,
                 kernel_size, kernel_init, use_bias):
        self.conv = nnx.Conv(input_features, output_features,
                             kernel_size=kernel_size,
                             padding="SAME",
                             use_bias=use_bias,
                             kernel_init=kernel_init,
                             rngs=rngs)

    def __call__(self, x):
        return nnx.max_pool(self.conv(x), window_shape=(2, 2),
                            strides=(2, 2), padding="SAME")


class CNN(nnx.Module):
    """Sequential convolutional PC model with a fully connected readout."""
    def __init__(self, features: Sequence[int], input_shape: Sequence[int],
                 output_shape: Sequence[int], *, rngs,
                 kernel_size: int | tuple[int, int] = 3,
                 kernel_init_scale: float = 1.0,
                 activation_fn: Callable = nnx.tanh,
                 use_bias = True):
        if len(input_shape) != 3:
            raise ValueError("input_shape must be channels-last (height, width, channels)")
        if not features:
            raise ValueError("features must include the input channel count")
        if input_shape[-1] != features[0]:
            raise ValueError("input_shape channels must equal features[0]")
        if isinstance(output_shape, int):
            output_shape = (output_shape,)

        height, width, _ = input_shape
        kernel_init = lecun_norm_w_scale(scale=kernel_init_scale)
        conv_layers = []
        conv_activation_shapes = []
        for fin, fout in zip(features[:-1], features[1:]):
            conv_layers.append(_ConvPool(fin, fout,
                                         kernel_size=kernel_size,
                                         use_bias=use_bias,
                                         kernel_init=kernel_init,
                                         rngs=rngs))
            # SAME stride-two pooling rounds odd spatial dimensions up.
            height = (height + 1) // 2
            width = (width + 1) // 2
            conv_activation_shapes.append((height, width, fout))
        readout_features = height * width * features[-1]
        output_features = int(jnp.prod(jnp.array(output_shape)))
        self.layers = [*conv_layers,
                       _ConvReadout(readout_features, output_features,
                                    use_bias=use_bias,
                                    kernel_init=kernel_init,
                                    rngs=rngs)]
        self.use_bias = use_bias
        self.activations = [Activation(jrng.normal(rngs.activations(), shape))
                            for shape in conv_activation_shapes]
        self.activations.append(Activation(jrng.normal(rngs.activations(), output_shape)))
        self.activation_fn = activation_fn

    def __call__(self, x, rclamp = None):
        if rclamp is not None:
            self.activations[-1].value = rclamp
        rs = [x, *(self.activation_fn(v.value) for v in self.activations[:-1])]
        vs = [layer(r) for layer, r in zip(self.layers, rs)]
        self.sow(nnx.Intermediate, "activations_ff", vs,
                 init_fn=(lambda: []),
                 reduce_fn=(lambda _, new: new))
        return vs[-1]

    def get_activations(self):
        return [act.value for act in self.activations]

    def set_activations(self, activations):
        for i, act in enumerate(activations):
            self.activations[i].value = act
