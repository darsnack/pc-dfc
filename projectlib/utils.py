import jax.numpy as jnp
import jax.tree as jtu
from flax import nnx

from projectlib.models import Activation

def noop(x, *_):
    return x

@nnx.jit
def accuracy(ypred, ytrue):
    return jnp.mean(jnp.argmax(ypred, axis=-1) == jnp.argmax(ytrue, axis=-1))

@nnx.jit
def get_model_params(model):
    """Extract all model parameters as a flat dictionary."""
    # Extract weights and biases from each layer
    filter = nnx.All(nnx.Param, nnx.Not(Activation))
    params = {k: v.value.copy()
              for k, v in nnx.variables(model, filter).flat_state()}

    return params

@nnx.jit
def compute_param_change_norm(old_params, new_params):
    """Compute the norm of parameter changes between two parameter sets."""
    # Compute squared differences for each parameter
    norm = jnp.sqrt(jtu.reduce(
        lambda x, y: x + y,
        jtu.map(lambda old, new: jnp.sum((new - old) ** 2), old_params, new_params)
    ))

    return norm
