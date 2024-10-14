from flax import nnx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax


class Model(nnx.Module):
    def __init__(self, intermediate_features: int = 16, rngs=nnx.Rngs):
        self.upscale = nnx.ConvTranspose(
            in_features=3,
            out_features=intermediate_features,
            kernel_size=(4, 4),
            padding=2,
            strides=(2, 2),
            rngs=rngs
        )
        self.deepen = nnx.Conv(
            in_features=intermediate_features,
            out_features=3,
            kernel_size=(3, 3),
            padding=1,
            rngs=rngs
        )

    def __call__(self, x: jax.Array):
        return self.deepen(self.upscale(x))


@jax.jit
def apply_model(state: TrainState, X: jax.Array, Y: jax.Array):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        preds = state.apply_fn(params, X)
        loss = jnp.mean(optax.squared_error(preds, Y))
        return loss, preds

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, preds), grads = grad_fn(state.params)
    return grads, loss


@jax.jit
def update_model(state: TrainState, grads):
    return state.apply_gradients(grads=grads)
