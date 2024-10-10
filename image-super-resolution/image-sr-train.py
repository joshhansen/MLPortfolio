import os

from absl import logging
import flax.linen as nn
from flax import nnx
from flax.training.train_state import TrainState
import imageio
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


# @dataclass
# class Params:
#     convs: list[nn.Conv]

#     @classmethod
#     def initialize(cls, layers: int = 8, features: int = 16):
#         convs = list()

#         for _ in range(layers):
#             convs.append(nn.Conv(features=features,
#                          kernel_size=(3,), padding='VALID', strides=2, kernel_dilation=2))

#         return cls(convs)
#         # emb_key, attn_key, attn_query_key, linear1_key, linear2_key = jrand.split(key, 5)
#         # emb =  initializer(emb_key, (vocab_len, EMBEDDING_DIMS), dtype=fX)
#         # attn =  MultiheadAttentionParams.initialize(attn_key, initializer, ATTN_HEADS, EMBEDDING_DIMS, ATTN_DIMS_PER_HEAD, EMBEDDING_DIMS)
#         # attn_query = initializer(attn_query_key, (target_len, EMBEDDING_DIMS), dtype=fX)
#         # linear1 =  Linear.initialize(linear1_key, MODEL_DIMS, MODEL_DIMS // 2, fX)
#         # linear2 =  Linear.initialize(linear2_key, MODEL_DIMS // 2, 1, fX)

#         # return cls(emb, attn, attn_query, linear1, linear2)


# def model(params: Params, full: jax.Array, small: jax.Array):
#     pass


# model_grad = jax.grad(model)


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


def train_epoch(
    state: TrainState,
    X: jax.Array,
    Y: jax.Array,
    rng: jax.Array
):
    # """Train for a single epoch."""
    # train_ds_size = len(train_ds['image'])
    # steps_per_epoch = train_ds_size // batch_size

    # perms = jax.random.permutation(rng, len(train_ds['image']))
    # perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    # perms = perms.reshape((steps_per_epoch, batch_size))

    # epoch_loss = []
    # epoch_accuracy = []

    # for perm in perms:
    #   batch_images = train_ds['image'][perm, ...]
    #   batch_labels = train_ds['label'][perm, ...]
    #   grads, loss, accuracy = apply_model(state, batch_images, batch_labels)
    #   state = update_model(state, grads)
    #   epoch_loss.append(loss)
    #   epoch_accuracy.append(accuracy)
    # train_loss = np.mean(epoch_loss)
    # train_accuracy = np.mean(epoch_accuracy)
    # return state, train_loss, train_accuracy

    # epoch_loss = list()

    grads, loss = apply_model(state, X, Y)
    state = update_model(state, grads)
    # epoch_loss.append(loss)

    # train_loss = jnp.mean(epoch_loss)

    # return state, train_loss
    return state, loss


@nnx.jit
def train_step(
    m: Model,
    opt: nnx.Optimizer,
    small: jax.Array,
    full: jax.Array
):
    def loss_fn(m: Model):
        pred = m(small)
        return jnp.mean(optax.squared_error(pred, full))

    loss, grads = nnx.value_and_grad(loss_fn)(m)
    opt.update(grads)  # in-place updates

    return loss


if __name__ == "__main__":
    HOME = os.environ['HOME']
    DATA_DIR = os.path.join(
        os.environ['HOME'], "Data/com/github/cvdfoundation/google-landmark")
    SMALL_DIR = os.path.join(DATA_DIR, "train_downsampled")
    FULL_DIR = os.path.join(DATA_DIR, "train")

    LR = 0.001
    MOMENTUM = 0.1
    ITS = 100

    rngs = nnx.Rngs(98239)

    m = Model(rngs=rngs)
    opt = nnx.Optimizer(m, optax.adam(1e-3))

    for epoch in range(ITS):
        count = 0
        total_train_loss = 0.0
        for dirpath, _dirnames, filenames in os.walk(SMALL_DIR):
            reldirpath = os.path.relpath(dirpath, SMALL_DIR)
            fulldirpath = os.path.join(FULL_DIR, reldirpath)
            print(f"dirpath: {dirpath}")
            print(reldirpath)
            print(fulldirpath)

            for filename in filenames:
                small_path = os.path.join(dirpath, filename)
                full_path = os.path.join(fulldirpath, filename)

                small_np = imageio.v3.imread(small_path)
                full_np = imageio.v3.imread(full_path)

                small: jax.Array = jnp.asarray(
                    small_np, dtype='float32').reshape(1, *small_np.shape)
                full: jax.Array = jnp.asarray(
                    full_np, dtype='float32').reshape(1, *full_np.shape)

                print(f"Small shape: {small.shape}")
                print(f"Full shape: {full.shape}")

                count += 1
                total_train_loss += train_step(m, opt, small, full)
                test_loss = -1

                if count % 10 == 0:

                    epoch_avg_train_loss = total_train_loss / count
                    print(
                        'epoch:% 3d, epoch avg train loss: %.4f, test_loss: %.4f,'
                        % (
                            epoch,
                            epoch_avg_train_loss,
                            test_loss,
                        )
                    )
