# Python stdlib imports
import argparse
from collections import deque
import math
import os
from pathlib import Path
import sys

# JAX-y imports
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
from model import Model, apply_model, erase_and_create_empty, INTERMEDIATE_FEATS


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
def loss(pred: jax.Array, large: jax.Array) -> jax.Array:
    return jnp.mean(optax.squared_error(pred, large))


@nnx.jit
def train_step(
    m: Model,
    opt: nnx.Optimizer,
    small: jax.Array,
    large: jax.Array
):
    def loss_fn(m: Model):
        pred = m(small)
        return loss(pred, large)

    l, grads = nnx.value_and_grad(loss_fn)(m)

    opt.update(grads)  # in-place updates

    return l

def data_part(filename: str) -> str:
    basename = Path(filename).stem
    if basename.endswith('0'):
        return 'test'
    if basename.endswith('1'):
        return 'valid'
    return 'train'

def mean(seq):
    if len(seq) == 0:
        return math.nan

    total = 0.0

    for x in seq:
        total += x

    return total / len(seq)


if __name__ == "__main__":
    # h = guppy.hpy()

    parser = argparse.ArgumentParser(
        prog='Image super-resolution trainer',
        description='Trains an image sr model')
    parser.add_argument("small_dir")
    parser.add_argument("large_dir")
    parser.add_argument('output_dir')
    parser.add_argument('factor', type=int)
    parser.add_argument('gpu', type=int)

    args = parser.parse_args()

    print(args)

    FACTOR=args.factor
    DEVICE=jax.devices()[args.gpu]
    BATCH=10
    DTYPE='float16'

    print(DEVICE)
    
    SMALL_CROP_WIDTH=700
    SMALL_CROP_HEIGHT=700
    LARGE_CROP_WIDTH=FACTOR*SMALL_CROP_WIDTH
    LARGE_CROP_HEIGHT=FACTOR*SMALL_CROP_HEIGHT

    HOME = os.environ['HOME']
    # DATA_DIR = args.training_images_dir
    # DATA_DIR = os.path.join(
    #     os.environ['HOME'], "Data/com/github/cvdfoundation/google-landmark")
    # SMALL_DIR = os.path.join(DATA_DIR, "train_downsampled")
    # FULL_DIR = os.path.join(DATA_DIR, "train")

    # print(f"DATA_DIR: {DATA_DIR}")

    SMALL_DIR = args.small_dir
    LARGE_DIR = args.large_dir

    LR = 1e-4
    ITS = 100
    RECENT = 100

    rngs = nnx.Rngs(98239)
    # mesh = jax.make_mesh((4,), jax.sharding.PartitionSpec('path'))
    # sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('path'))

    m = Model(rngs, FACTOR)
    opt = nnx.Optimizer(m, optax.adam(LR))
    with ocp.CheckpointManager(
        erase_and_create_empty(args.output_dir),
        options=ocp.CheckpointManagerOptions(
            # best_fn=lambda metrics: metrics['test_loss'],
            # best_mode='min',
            max_to_keep=3,
            save_interval_steps=2
        ),
    ) as checkpoint_mgr:
        for epoch in range(ITS):
            count = 0
            train_count = 0
            test_count = 0
            img_load_errors = 0
            total_train_loss = 0.0
            total_test_loss = 0.0

            recent_train_losses: deque[float] = deque()
            recent_test_losses: deque[float] = deque()

            batch_train_small: list[jax.Array] = list()
            batch_train_large: list[jax.Array] = list()
            batch_test_small: list[jax.Array] = list()
            batch_test_large: list[jax.Array] = list()

            for dirpath, _dirnames, filenames in os.walk(SMALL_DIR):
                reldirpath = os.path.relpath(dirpath, SMALL_DIR)
                largedirpath = os.path.join(LARGE_DIR, reldirpath)

                for filename in filenames:
                    count += 1

                    part = data_part(filename)                    

                    small_path = os.path.join(dirpath, filename)
                    large_path = os.path.join(largedirpath, filename)

                    print(small_path)

                    try:
                        small_np = imageio.v3.imread(small_path, mode="RGB")
                    except Exception:
                        img_load_errors += 1
                        sys.stderr.write(f"Couldn't load {small_path}\n")
                        continue

                    try:
                        large_np = imageio.v3.imread(large_path, mode="RGB")
                    except Exception:
                        img_load_errors += 1
                        continue

                    if small_np.shape[0] < SMALL_CROP_WIDTH or small_np.shape[1] < SMALL_CROP_HEIGHT:
                        continue

                    small: jax.Array = jnp.asarray(small_np, dtype=DTYPE, device=DEVICE)
                    large: jax.Array = jnp.asarray(large_np, dtype=DTYPE, device=DEVICE)

                    del small_np
                    del large_np

                    small_new_shape = (SMALL_CROP_WIDTH, SMALL_CROP_HEIGHT, 3)
                    large_new_shape = (LARGE_CROP_WIDTH, LARGE_CROP_HEIGHT, 3)

                    small = jnp.resize(small, small_new_shape)
                    large = jnp.resize(large, large_new_shape)

                    if part == 'train':
                        batch_train_small.append(small)
                        batch_train_large.append(large)
                    elif part == 'test':
                        batch_test_small.append(small)
                        batch_test_large.append(large)

                    checkpoint = False
                    if len(batch_train_small) >= BATCH:
                        X = jnp.stack(batch_train_small)
                        Y = jnp.stack(batch_train_large)
                        batch_train_small = list()
                        batch_train_large = list()

                        # X = jax.device_put(X, sharding)
                        # Y = jax.device_put(Y, sharding)

                        train_count += BATCH

                        l = train_step(m, opt, X, Y)
                        total_train_loss += l

                        recent_train_losses.append(l / BATCH)
                        if len(recent_train_losses) > RECENT:
                            recent_train_losses.popleft()

                        checkpoint = True
                    elif len(batch_test_small) >= BATCH:
                        X = jnp.stack(batch_test_small)
                        Y = jnp.stack(batch_test_large)
                        batch_test_small = list()
                        batch_test_large = list()
                        
                        
                        # X = jax.device_put(X, sharding)
                        # Y = jax.device_put(Y, sharding)

                        test_count += BATCH

                        pred = m(X)

                        l = loss(pred, Y)
                        total_test_loss += l

                        recent_test_losses.append(l / BATCH)
                        if len(recent_test_losses) > RECENT:
                            recent_test_losses.popleft()
                        
                        checkpoint = True

                    if checkpoint:
                        state = nnx.state(m)
                        checkpoint_mgr.save(
                            train_count,
                            args=ocp.args.StandardSave(state),
                            # metrics={
                            #     'test_loss': epoch_avg_test_loss,
                            # }
                        )

                        # jax.profiler.save_device_memory_profile("/tmp/memory.prof")

                    if count % 10 == 0 and count > 0:
                        try:
                            epoch_avg_train_loss = total_train_loss / train_count
                        except ZeroDivisionError:
                            epoch_avg_train_loss = float('nan')
                            
                        try:
                            epoch_avg_test_loss = total_test_loss / test_count
                        except ZeroDivisionError:
                            epoch_avg_test_loss = float('nan')

                        recent_train_loss = mean(recent_train_losses)
                        recent_test_loss = mean(recent_test_losses)

                        print(total_train_loss)
                        print(total_test_loss)

                        print(
                            'epoch:% 3d, train_count: %d, test_count: %d, avg train loss: %.4f, recent train loss: %.4f, avg test_loss: %.4f, recent test loss: %.4f, imloaderrs: %d'
                            % (
                                epoch,
                                train_count,
                                test_count,
                                epoch_avg_train_loss,
                                recent_train_loss,
                                epoch_avg_test_loss,
                                recent_test_loss,
                                img_load_errors
                            )
                        )
                        sys.stdout.flush()

                        # if count % 100 == 0 and train_count > 0:
                        #     # Prevent JIT compilation caches from growing without end
                        #     jax.clear_caches()

                        # print(h.heap())

