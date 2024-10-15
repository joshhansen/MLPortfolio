from pathlib import Path
import shutil

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
        self.embedders = [
            nnx.Conv(
                in_features=3,
                out_features=3,
                kernel_size=(3,3),
                padding=0,
                rngs=rngs,
            )
            for i in range(30)
        ]

    def __call__(self, x: jax.Array):
        # Main line of dilated convolution plus regular conv layer(s)
        out = self.deepen(self.upscale(x))

        # Generate a 1x4x4x3 image embedding
        rescaled = jax.image.resize(x, (1, 64, 64, 3), "linear")
        embed = rescaled
        for emb in self.embedders:
            embed = emb(embed)

        # Convolve the main line output by the embedding
        return jax.scipy.signal.convolve(out, embed, mode='same', method='direct')
        


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

def erase_and_create_empty(path: str) -> Path:
    p = Path(path)

    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)

    p.mkdir(parents=True)

    return p
