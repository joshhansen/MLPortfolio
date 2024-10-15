import argparse
import os
from pathlib import Path
import shutil
import sys

import flax
from flax import nnx
from flax.training.train_state import TrainState
import imageio
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import orbax.checkpoint as ocp

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


def erase_and_create_empty(path: str) -> Path:
    p = Path(path)

    if p.is_dir():
        shutil.rmtree(p)
    else:
        p.unlink(missing_ok=True)

    p.mkdir(parents=True)

    return p


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog='Image super-resolution model run',
        description='Applies a trained super-resolution model to images')

    parser.add_argument("checkpoints_path", type=Path)
    # parser.add_argument("epoch", type=int)
    parser.add_argument("image_paths", type=Path, nargs='+')

    args = parser.parse_args()

    # rngs = nnx.Rngs(98239)

    # m = Model(rngs=rngs)
    # opt = nnx.Optimizer(m, optax.adam(1e-3))
    with ocp.CheckpointManager(
        args.checkpoints_path.absolute(),
        options=ocp.CheckpointManagerOptions(max_to_keep=3, save_interval_steps=2),
    ) as checkpoint_mgr:

        epoch = checkpoint_mgr.latest_step()
        #  restore_args = ocp.checkpoint_utils.construct_restore_args(abstract_pytree),
        #  state = checkpoint_mgr.restore(
        #     epoch,
        #     items=abstract_pytree,
        #     restore_kwargs={'restore_args': restore_args}
        # )


        abstract_pytree = jax.tree_util.tree_map(
            ocp.utils.to_shape_dtype_struct, nnx.state(Model(rngs=nnx.Rngs(439482434)))
        )
        state = checkpoint_mgr.restore(epoch, args=ocp.args.StandardRestore(abstract_pytree))

        print(state)

        abstract_model = nnx.eval_shape(lambda: Model(rngs=nnx.Rngs(382931289)))
        gd = nnx.graphdef(abstract_model)

        print(gd)

        m = nnx.merge(gd, state)

        print(m)

    outdir = erase_and_create_empty('/tmp/image-sr-output')

    for i, path in enumerate(args.image_paths):
        try:
            small_np = imageio.v3.imread(path, mode="RGB")
        except Exception:
            sys.stderr.write(f"Couldn't load {path}\n")
            continue

        small: jax.Array = jnp.asarray(
            small_np, dtype='float32').reshape(1, *small_np.shape)
        
        # pred = m(small)

        pred = small

        new_shape = list(pred.shape)[1:]

        pred = pred.reshape(*new_shape)

        # Back to Numpy
        pred = onp.array(pred)

        outpath = outdir / f"{i}.png"

        with open(outpath, 'wb') as w:
            imageio.v3.imwrite(outpath, pred, mode="RGB")

        # for epoch in range(ITS):
        #     train_count = 0
        #     test_count = 0
        #     img_load_errors = 0
        #     total_train_loss = 0.0
        #     total_test_loss = 0.0
        #     for dirpath, _dirnames, filenames in os.walk(SMALL_DIR):
        #         reldirpath = os.path.relpath(dirpath, SMALL_DIR)
        #         fulldirpath = os.path.join(FULL_DIR, reldirpath)

        #         for filename in filenames:

        #             if train_count % 10 == 0 and train_count > 0:

        #                 # checkpoint_manager.save(epoch, args=ocp.args.Composite(
        #                 #     state=ocp.args.StandardSave(m),
        #                 #     # extra_params=ocp.args.JsonSave(extra_params),
        #                 # ))
        #                 state = nnx.state(m)
        #                 checkpoint_mgr.save(epoch, state)

        #                 # state = nnx.state(m)
        #                 # ser = flax.serialization.to_bytes(state)

        #                 # with open('./model.ser', 'wb') as w:
        #                 #     w.write(ser)
