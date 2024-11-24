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
from flax.typing import Dtype
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import optax
import orbax.checkpoint as ocp

# Other 3rd party imports
# import guppy
import imageio

jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# Self imports
from model import Model, erase_and_create_empty, INTERMEDIATE_FEATS, assert_num, assert_arr_num

def assert_some_nonzero(x: jax.Array):
    assert(jnp.count_nonzero(x).item() > 0)


@nnx.jit
def loss(pred: jax.Array, large: jax.Array) -> jax.Array:
    # print(f"pred.shape: {pred.shape}")
    # print(f"large.shape: {large.shape}")

    # print(f"pred: {pred}")
    # print(f"large: {large}")

    return jnp.mean(optax.squared_error(pred, large))


@nnx.jit
def train_step(
    m: Model,
    opt: nnx.Optimizer,
    small: jax.Array,
    large: jax.Array
):
    # print(f"train_step small.shape: {small.shape}")
    # print(f"train_step large.shape: {large.shape}")
    # assert_arr_num(small)
    # assert_arr_num(large)
    
    def loss_fn(m: Model):
        pred = m(small)
        return loss(pred, large)

    l, grads = nnx.value_and_grad(loss_fn)(m)

    # print(f"grads devs: {grads.devices()}")

    # for x in (grads.deep, grads.deeper, grads.deepest):
    #     assert_arr_num(x.kernel.value)
    #     assert_some_nonzero(x.kernel.value)


    opt.update(grads)  # in-place updates

    return l

def data_part(filename: str) -> str:
    basename = Path(filename).stem

    last = basename[-1]

    value = int(f"0x{last}", 16)

    if value == 0:
        return 'valid'

    if value < 5:
        return 'test'

    return 'train'

def mean(seq):
    if len(seq) == 0:
        return math.nan

    total = 0.0

    for x in seq:
        total += x

    return total / len(seq)

@nnx.jit
def create_sharded_model(rngs: nnx.Rngs):
    m = Model(rngs)
    state = nnx.state(m)                   # The m's state, a pure pytree.
    pspecs = nnx.get_partition_spec(state)     # Strip out the annotations from state.
    sharded_state = jax.lax.with_sharding_constraint(state, pspecs)
    nnx.update(m, sharded_state)           # The m is sharded now!
    return m

if __name__ == "__main__":
    # h = guppy.hpy()

    parser = argparse.ArgumentParser(
        prog='Image super-resolution trainer',
        description='Trains an image sr model')
    parser.add_argument("small_dir")
    parser.add_argument("large_dir")
    parser.add_argument('output_dir')
    parser.add_argument('factor', type=int)

    args = parser.parse_args()

    print(args)

    # with jax.experimental.enable_x64():
    load_device = jax.devices('gpu')[0]
    compute_devices = jax.devices('gpu')[1:]

    FACTOR=args.factor

    # Make sure we're a multiple of # of devices
    # This lets us by the batch dimension
    TRAIN_BATCH=5 * len(compute_devices)
    TEST_BATCH=len(compute_devices)
    # TRAIN_BATCH=10
    # TEST_BATCH=10
    DTYPE: Dtype = 'float32'
    PARAM_DTYPE: Dtype = 'float32'

    SMALL_CROP_WIDTH=700
    SMALL_CROP_HEIGHT=700
    LARGE_CROP_WIDTH=FACTOR*SMALL_CROP_WIDTH
    LARGE_CROP_HEIGHT=FACTOR*SMALL_CROP_HEIGHT

    SMALL_DIR = args.small_dir
    LARGE_DIR = args.large_dir

    LR = 1e-6
    ITS = 100
    RECENT = 100

    rngs = nnx.Rngs(98239)
    key = jax.random.key(2892)

    print(f"Load device: {load_device}")
    # print(f"Compute devices: {compute_devices}")

    with Mesh(
        devices=compute_devices,
        axis_names=('gpu',),
    ) as mesh:
        # (batch, x, y, channels)
        # Shards batches across GPUs
        data_sharding = NamedSharding(mesh, PartitionSpec('gpu'))

        m = create_sharded_model(rngs)
        # m = Model(rngs, param_dtype=PARAM_DTYPE)

        print(m.deep.kernel.sharding)
        print(m.deeper.kernel.sharding)
        print(m.deepest.kernel.sharding)

        print(m.deep.kernel.shape)
        print(m.deeper.kernel.shape)
        print(m.deepest.kernel.shape)

        print(m.deep.kernel.dtype)
        print(m.deeper.kernel.dtype)
        print(m.deepest.kernel.dtype)

        assert_arr_num(m.deep.kernel.value)
        assert_arr_num(m.deeper.kernel.value)
        assert_arr_num(m.deepest.kernel.value)

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

                min_recent_test_loss = float('inf')

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

                        # Convert to Jax arrays and normalize small to [0, 1]
                        small: jax.Array = jnp.asarray(small_np, dtype=DTYPE, device=load_device)
                        large: jax.Array = jnp.asarray(large_np, dtype=DTYPE, device=load_device) / 255.0

                        del small_np
                        del large_np

                        # key, subkey = jax.random.split(key)
                        # large = jax.random.uniform(subkey, (LARGE_CROP_WIDTH, LARGE_CROP_HEIGHT, 3), dtype=DTYPE) * 255
                        # small = jax.image.resize(large, (SMALL_CROP_WIDTH, SMALL_CROP_HEIGHT, 3), 'nearest')

                        # print(f"small dtype: {small.dtype}")
                        # print(f"large dtype: {large.dtype}")

                        assert_arr_num(small)
                        assert_arr_num(large)
                        assert_some_nonzero(small)
                        assert_some_nonzero(large)

                        small_new_shape = (SMALL_CROP_WIDTH, SMALL_CROP_HEIGHT, 3)
                        large_new_shape = (LARGE_CROP_WIDTH, LARGE_CROP_HEIGHT, 3)

                        small = jnp.resize(small, small_new_shape)
                        large = jnp.resize(large, large_new_shape)

                        assert_arr_num(small)
                        assert_arr_num(large)

                        assert_some_nonzero(small)
                        assert_some_nonzero(large)

                        # print(f"loading small: {small}")
                        # print(f"loading large: {large}")

                        if part == 'train':
                            batch_train_small.append(small)
                            batch_train_large.append(large)
                        elif part == 'test':
                            batch_test_small.append(small)
                            batch_test_large.append(large)

                        checkpoint = False
                        if len(batch_train_small) >= TRAIN_BATCH:
                            assert(len(batch_train_small) == len(batch_train_large))

                            # X_arr = jnp.asarray(batch_train_small, dtype=DTYPE)
                            # Y_arr = jnp.asarray(batch_train_small, dtype=DTYPE)

                            # print(f"X_arr shape: {X_arr.shape}")
                            # print(f"Y_arr shape: {Y_arr.shape}")

                            # X_shards = jnp.array_split(X_arr, len(compute_devices))
                            # Y_shards = jnp.array_split(Y_arr, len(compute_devices))

                            # # print(f"X_shards shape: {X_shards.shape}")
                            # # print(f"Y_shards shape: {Y_shards.shape}")

                            # X = jax.device_put_sharded(X_shards, compute_devices).block_until_ready()
                            # Y = jax.device_put_sharded(Y_shards, compute_devices).block_until_ready()
                        
                            X = jnp.stack(batch_train_small)
                            Y = jnp.stack(batch_train_large)
                            batch_train_small = list()
                            batch_train_large = list()

                            assert_arr_num(X)
                            assert_arr_num(Y)
                            assert_some_nonzero(X)
                            assert_some_nonzero(Y)

                            # print(f"X initial: {X}")
                            # print(f"Y initial: {Y}")
                            # print(f"X shape: {X.shape}")
                            # print(f"Y shape: {Y.shape}")
                            # print(f"X sharding: {X.sharding}")
                            # print(f"Y sharding: {Y.sharding}")
                            # print(f"X devices: {X.devices()}")
                            # print(f"Y devices: {Y.devices()}")
                            # print(f"X dtype: {X.dtype}")
                            # print(f"Y dtype: {Y.dtype}")

                            X = jax.device_put(X, data_sharding)#.block_until_ready()#, donate=True)
                            Y = jax.device_put(Y, data_sharding)#.block_until_ready()#, donate=True)

                            # print(f"X after put: {X}")
                            # print(f"Y after put: {Y}")
                            # print(f"X shape: {X.shape}")
                            # print(f"Y shape: {Y.shape}")
                            # print(f"X sharding: {X.sharding}")
                            # print(f"Y sharding: {Y.sharding}")
                            # print(f"X devices: {X.devices()}")
                            # print(f"Y devices: {Y.devices()}")
                            # print(f"X dtype: {X.dtype}")
                            # print(f"Y dtype: {Y.dtype}")
                        
                            assert_arr_num(X)
                            assert_arr_num(Y)
                            assert_some_nonzero(X)
                            assert_some_nonzero(Y)


                            train_count += TRAIN_BATCH

                            l = train_step(m, opt, X, Y).item()

                            print(f"loss: {l}")

                            assert_num(l)
                    
                            total_train_loss += l

                            recent_train_losses.append(l / TRAIN_BATCH)
                            if len(recent_train_losses) > RECENT:
                                recent_train_losses.popleft()

                            checkpoint = True

                            del X
                            del Y
                        elif len(batch_test_small) >= TEST_BATCH:
                            assert(len(batch_test_small) == len(batch_test_large))
                            X = jnp.stack(batch_test_small)
                            Y = jnp.stack(batch_test_large)
                            batch_test_small = list()
                            batch_test_large = list()

                            assert_arr_num(X)
                            assert_arr_num(Y)
                
                            X = jax.device_put(X, data_sharding)#, donate=True)
                            Y = jax.device_put(Y, data_sharding)#, donate=True)

                            # assert_arr_num(X)
                            # assert_arr_num(Y)

                            # print(f"X sharding: {X.sharding}")
                            # print(f"Y sharding: {Y.sharding}")
                            # print(f"X devices: {X.devices()}")
                            # print(f"Y devices: {Y.devices()}")
                            # print(f"X dtype: {X.dtype}")
                            # print(f"Y dtype: {Y.dtype}")

                            assert_arr_num(X)
                            assert_arr_num(Y)
                            assert_some_nonzero(X)
                            assert_some_nonzero(Y)

                            test_count += TEST_BATCH

                            pred = m(X)

                            assert_arr_num(pred)

                            l = loss(pred, Y).item()

                            assert_num(l)

                            total_test_loss += l

                            recent_test_losses.append(l / TEST_BATCH)
                            if len(recent_test_losses) > RECENT:
                                recent_test_losses.popleft()
                
                            checkpoint = True

                            del X
                            del Y

                        assert_arr_num(m.deep.kernel.value)
                        assert_arr_num(m.deeper.kernel.value)
                        assert_arr_num(m.deepest.kernel.value)
                
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

                        if checkpoint and recent_test_loss <= min_recent_test_loss:
                            state = nnx.state(m)
                            checkpoint_mgr.save(
                                train_count,
                                args=ocp.args.StandardSave(state),
                                # metrics={
                                #     'test_loss': epoch_avg_test_loss,
                                # }
                            )

                            jax.profiler.save_device_memory_profile("/tmp/memory.prof")

                        if recent_test_loss < min_recent_test_loss:
                            min_recent_test_loss = recent_test_loss

                        if count % 10 == 0 and count > 0:

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

