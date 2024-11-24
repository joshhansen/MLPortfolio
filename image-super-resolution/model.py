import math
from pathlib import Path
import shutil

from flax import nnx
from flax.typing import Dtype
from jax.sharding import PartitionSpec
from flax.training.train_state import TrainState
import jax
import jax.nn as jnn
import jax.numpy as jnp
import optax


INTERMEDIATE_FEATS=16

def assert_num(x):
    assert(not math.isnan(x))
    assert(not math.isinf(x))

def assert_arr_num(x: jax.Array):
    assert(not jnp.isnan(x).any())
    assert(not jnp.isinf(x).any())

class Model(nnx.Module):
    def __init__(self, rngs:nnx.Rngs, param_dtype: Dtype = 'float64', use_bias = False):
        init_fn = nnx.initializers.lecun_normal()
        # init_fn = nnx.initializers.constant(0.5)

        # Distribute the model to all devices
        partitioning = (None,)
        self.deep = nnx.Conv(
            in_features=3,
            out_features=INTERMEDIATE_FEATS,
            kernel_size=(7, 7),
            padding='SAME',
            rngs=rngs,
            use_bias=use_bias,
            param_dtype=param_dtype,
            kernel_init=nnx.with_partitioning(init_fn, partitioning),
        )
        self.deeper = nnx.Conv(
            in_features=INTERMEDIATE_FEATS,
            out_features=INTERMEDIATE_FEATS,
            kernel_size=(5, 5),
            padding='SAME',
            rngs=rngs,
            param_dtype=param_dtype,
            use_bias=use_bias,
            kernel_init=nnx.with_partitioning(init_fn, partitioning),
        )
        self.deepest = nnx.Conv(
            in_features=INTERMEDIATE_FEATS,
            out_features=3,
            kernel_size=(3, 3),
            padding='SAME',
            rngs=rngs,
            param_dtype=param_dtype,
            use_bias=use_bias,
            kernel_init=nnx.with_partitioning(init_fn, partitioning),
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
        # assert_arr_num(x)

        # print(f"x shape: {x.shape}")
        
        new_shape = (x.shape[0], x.shape[1] * 2,
                     x.shape[2] * 2, 3)
        upscaled = jax.image.resize(x, new_shape, "nearest")

        # assert_arr_num(upscaled)

        # print(f"upscaled shape: {upscaled.shape}")

        # assert_arr_num(self.deep.kernel.value)
        out = self.deep(upscaled)
        # assert_arr_num(out)
        # out = jnn.sigmoid(out)
        # assert_arr_num(out)

        # assert_arr_num(self.deeper.kernel)
        out = self.deeper(out)
        # assert_arr_num(out)
        # out = jnn.sigmoid(out)
        # assert_arr_num(out)

        # assert_arr_num(self.deepest.kernel)
        out = self.deepest(out)
        # assert_arr_num(out)
        # out = jnn.sigmoid(out) * 255
        # assert_arr_num(out)

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

        return out * 255.0


def erase_and_create_empty(path: str) -> Path:
    p = Path(path)

    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)

    p.mkdir(parents=True)

    return p
