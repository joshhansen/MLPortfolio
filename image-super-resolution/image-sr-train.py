import os
from pathlib import Path
import sys

import flax
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
    grads, loss = apply_model(state, X, Y)
    state = update_model(state, grads)

    return state, loss


@jax.jit
def loss(pred: jax.Array, full: jax.Array) -> jax.Array:
    return jnp.mean(optax.squared_error(pred, full))


@nnx.jit
def train_step(
    m: Model,
    opt: nnx.Optimizer,
    small: jax.Array,
    full: jax.Array
):
    def loss_fn(m: Model):
        pred = m(small)
        return loss(pred, full)

    l, grads = nnx.value_and_grad(loss_fn)(m)
    opt.update(grads)  # in-place updates

    return l


def is_testing(filename: str) -> bool:
    basename = Path(filename).stem
    return basename.endswith('0')


def is_valid(filename: str) -> bool:
    basename = Path(filename).stem
    return basename.endswith('1')


if __name__ == "__main__":
    HOME = os.environ['HOME']
    DATA_DIR = os.path.join(
        os.environ['HOME'], "Data/com/github/cvdfoundation/google-landmark")
    SMALL_DIR = os.path.join(DATA_DIR, "train_downsampled")
    FULL_DIR = os.path.join(DATA_DIR, "train")

    print(f"DATA_DIR: {DATA_DIR}")
    sys.stdout.flush()

    LR = 0.001
    MOMENTUM = 0.1
    ITS = 100

    rngs = nnx.Rngs(98239)

    m = Model(rngs=rngs)
    opt = nnx.Optimizer(m, optax.adam(1e-3))

    for epoch in range(ITS):
        train_count = 0
        test_count = 0
        total_train_loss = 0.0
        total_test_loss = 0.0
        for dirpath, _dirnames, filenames in os.walk(SMALL_DIR):
            reldirpath = os.path.relpath(dirpath, SMALL_DIR)
            fulldirpath = os.path.join(FULL_DIR, reldirpath)

            for filename in filenames:
                testing = is_testing(filename)
                valid = is_valid(filename)

                small_path = os.path.join(dirpath, filename)
                full_path = os.path.join(fulldirpath, filename)

                print(small_path)

                small_np = imageio.v3.imread(small_path, mode="RGB")
                full_np = imageio.v3.imread(full_path, mode="RGB")

                small: jax.Array = jnp.asarray(
                    small_np, dtype='float32').reshape(1, *small_np.shape)
                full: jax.Array = jnp.asarray(
                    full_np, dtype='float32').reshape(1, *full_np.shape)

                # print(f"Small shape: {small.shape}")
                # print(f"Full shape: {full.shape}")

                sys.stdout.flush()

                small_new_shape = list(small.shape)
                full_new_shape = list(full.shape)
                resize_small = False
                resize_full = False
                for dim in (1, 2):
                    assert full_new_shape[dim] > small_new_shape[dim]

                    # Make sure the larger dim is even
                    if full_new_shape[dim] % 2 == 1:
                        resize_full = True
                        full_new_shape[dim] -= 1

                    assert full_new_shape[dim] % 2 == 0

                    # Decrease the smaller dim until 2*smaller = larger
                    while small_new_shape[dim] * 2 > full_new_shape[dim]:
                        resize_small = True
                        small_new_shape[dim] -= 1

                    assert small_new_shape[dim] * 2 == full_new_shape[dim]

                if resize_small:
                    small = jnp.resize(small, tuple(small_new_shape))

                if resize_full:
                    full = jnp.resize(full, tuple(full_new_shape))

                # print(f"small_new_shape: {small_new_shape}")
                # print(f"full_new_shape: {full_new_shape}")

                if testing:
                    test_count += 1
                    pred = m(small)
                    total_test_loss += loss(pred, full)
                elif not valid:
                    train_count += 1
                    total_train_loss += train_step(m, opt, small, full)

                if train_count % 10 == 0 and train_count > 0:
                    epoch_avg_train_loss = total_train_loss / train_count
                    try:
                        epoch_avg_test_loss = total_test_loss / test_count
                    except ZeroDivisionError:
                        epoch_avg_test_loss = float('nan')

                    print(
                        'epoch:% 3d, train_count: %d, epoch avg train loss: %.4f, epoch avg test_loss: %.4f,'
                        % (
                            epoch,
                            train_count,
                            epoch_avg_train_loss,
                            epoch_avg_test_loss,
                        )
                    )

                    sys.stdout.flush()

                    state = nnx.state(m)
                    ser = flax.serialization.to_bytes(state)

                    with open('./model.ser', 'wb') as w:
                        w.write(ser)
