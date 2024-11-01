from pathlib import Path
import shutil

from flax import nnx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

DEEP = 3

INTERMEDIATE_FEATS = 16


class Model(nnx.Module):
    def __init__(self, rngs=nnx.Rngs):
        # self.upscale_rgb = nnx.ConvTranspose(
        #     in_features=3,
        #     out_features=3,
        #     kernel_size=(1, 3, 3),
        #     # kernel_dilation=2,
        #     # padding=0,
        #     padding='SAME',
        #     strides=(1, 2, 2),
        #     rngs=rngs
        # )
        self.deep = nnx.Conv(
            in_features=INTERMEDIATE_FEATS,
            out_features=INTERMEDIATE_FEATS,
            kernel_size=(7, 7),
            padding='SAME',
            rngs=rngs,
        )
        self.deeper = nnx.Conv(
            in_features=INTERMEDIATE_FEATS,
            out_features=INTERMEDIATE_FEATS,
            kernel_size=(5, 5),
            padding='SAME',
            rngs=rngs,
        )
        self.deepest = nnx.Conv(
            in_features=INTERMEDIATE_FEATS,
            out_features=3,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs,
        )
        # self.upscale_other = nnx.ConvTranspose(
        #     in_features=3,
        #     out_features=intermediate_features,
        #     kernel_size=(6, 6),
        #     padding=3,
        #     strides=(2, 2),
        #     rngs=rngs
        # )
        # self.deep = [
        #     nnx.Conv(
        #         in_features=intermediate_features,
        #         out_features=intermediate_features if i < DEEP-1 else 3,
        #         kernel_size=(3, 3),
        #         padding=1,
        #         rngs=rngs
        #     )
        #     for i in range(DEEP)
        # ]
        # self.embedder = nnx.MultiHeadAttention(
        #     num_heads=1,
        #     in_features=3,
        #     qkv_features=3,
        #     decode=False,
        #     rngs=rngs,
        # )

    def __call__(self, x: jax.Array):
        new_shape = (x.shape[0], x.shape[1] * 2,
                     x.shape[2] * 2, INTERMEDIATE_FEATS)
        upscaled = jax.image.resize(x, new_shape, "nearest")

        out = self.deep(upscaled)
        out = self.deeper(out)
        out = self.deepest(out)

        # Main line of dilated convolution plus regular conv layer(s)
        # out = self.upscale_rgb(x)
        # out = self.deep(out)

        # for d in self.deep:
        #     out = d(out)

        # Generate a 1x4x4x3 image embedding
        # rescaled = jax.image.resize(x, (1, 32, 32, 3), "linear")
        # embed = self.embedder(rescaled)
        # embed = rescaled
        # for emb in self.embedders:
        #     embed = emb(embed)

        # Convolve the main line output by the embedding
        # return jax.scipy.signal.convolve(out, embed, mode='same', method='direct')

        return out


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
