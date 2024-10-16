# Python stdlib imports
import argparse
from collections import deque
import os
from pathlib import Path
import shutil
import sys

#JAX-y imports
from flax import nnx
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp

# Other 3rd party imports
# import guppy
import imageio

# Self imports
from model import Model, apply_model, erase_and_create_empty


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


def mean(seq):
    total = 0.0

    for x in seq:
        total += x

    return total / len(seq)

if __name__ == "__main__":
    # h = guppy.hpy()
    
    parser = argparse.ArgumentParser(
        prog='Image super-resolution trainer',
        description='Trains an image sr model')
    parser.add_argument('output_dir')

    args = parser.parse_args()
    
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
    RECENT = 100

    rngs = nnx.Rngs(98239)

    m = Model(rngs=rngs)
    opt = nnx.Optimizer(m, optax.adam(1e-3))
    with ocp.CheckpointManager(
        erase_and_create_empty(args.output_dir),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2),
    ) as checkpoint_mgr:
        for epoch in range(ITS):
            train_count = 0
            test_count = 0
            img_load_errors = 0
            total_train_loss = 0.0
            total_test_loss = 0.0

            recent_train_losses: deque[float] = deque()
            recent_test_losses: deque[float] = deque()
            for dirpath, _dirnames, filenames in os.walk(SMALL_DIR):
                reldirpath = os.path.relpath(dirpath, SMALL_DIR)
                fulldirpath = os.path.join(FULL_DIR, reldirpath)

                for filename in filenames:
                    testing = is_testing(filename)
                    valid = is_valid(filename)

                    small_path = os.path.join(dirpath, filename)
                    full_path = os.path.join(fulldirpath, filename)

                    print(small_path)

                    try:
                        small_np = imageio.v3.imread(small_path, mode="RGB")
                    except Exception:
                        img_load_errors += 1
                        sys.stderr.write(f"Couldn't load {small_path}\n")
                        continue

                    try:
                        full_np = imageio.v3.imread(full_path, mode="RGB")
                    except Exception:
                        img_load_errors += 1
                        sys.stderr.write(f"Couldn't load {full_path}\n")
                        continue

                    small: jax.Array = jnp.asarray(small_np, dtype='float32').reshape(1, *small_np.shape)
                    full: jax.Array = jnp.asarray(full_np, dtype='float32').reshape(1, *full_np.shape)

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
                        l = loss(pred, full)
                        total_test_loss += l

                        recent_test_losses.append(l)
                        if len(recent_test_losses) > RECENT:
                            recent_test_losses.popleft()
                    elif not valid:
                        train_count += 1
                        l = train_step(m, opt, small, full)
                        total_train_loss += l

                        recent_train_losses.append(l)
                        if len(recent_train_losses) > RECENT:
                            recent_train_losses.popleft()

                    if train_count % 10 == 0 and train_count > 0:
                        epoch_avg_train_loss = total_train_loss / train_count
                        try:
                            epoch_avg_test_loss = total_test_loss / test_count
                        except ZeroDivisionError:
                            epoch_avg_test_loss = float('nan')

                        recent_train_loss = mean(recent_train_losses)
                        recent_test_loss = mean(recent_test_losses)

                        print(
                            'epoch:% 3d, train_count: %d, avg train loss: %.4f, recent train loss: %.4f, avg test_loss: %.4f, recent test loss: %.4f, imloaderrs: %d'
                            % (
                                epoch,
                                train_count,
                                epoch_avg_train_loss,
                                recent_train_loss,
                                epoch_avg_test_loss,
                                recent_test_loss,
                                img_load_errors
                            )
                        )

                        sys.stdout.flush()

                        if train_count % 100 == 0 and train_count > 0:
                            # Prevent JIT compilation caches from growing without end
                            jax.clear_caches()

                        # print(h.heap())

                        state = nnx.state(m)
                        checkpoint_mgr.save(epoch, args=ocp.args.StandardSave(state))
