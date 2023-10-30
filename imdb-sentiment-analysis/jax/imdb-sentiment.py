from imdb_sa_common import load

from typing import Mapping

from collections import Counter
import itertools
import json
import os
import re
import time


from jax import config
config.update("jax_debug_nans", True)

import jax
from jax import debug as jdbg
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrand
from jax import tree_util as jtree

from jax.experimental import checkify

from more_itertools import unzip

import optax

EMBEDDING_DIMS = 20
ATTN_DIMS = 16
ATTN_HEADS = 5
ATTN_SCALE = 1.0 / jnp.sqrt(ATTN_DIMS)
fX = jnp.float32
iX = jnp.uint32


def accuracy(preds, y):
 matching = y == preds

 correct = matching.sum()

 return correct / x_test.shape[0]

def model(params, x):
 # out = emb[x].mean(axis=1)
 out = params['emb'][x]

 # jdbg.print("embedded shape: {shape}", shape=out.shape)

 # jdbg.print("self-attn shape: {shape}", shape=params['self-attn'].shape)

 attns = [ (out @ attn).mean(axis=1) for attn in params['self-attn'] ]

 for attn in attns:
  jdbg.print("attn shape {shape}", shape=attn.shape)

 out = jnp.concatenate(attns, axis=1)

 jdbg.print("post stack shape: {shape}", shape=out.shape)

 out = out @ params['self-attn-stack']

 jdbg.print("post stackdot shape {shape}", shape=out.shape)
 
 # # jdbg.print(f"out.dtype {out.dtype} w.dtype {params[1]['w'].dtype} b.dtype {params[1]['b'].dtype}")

 # out = out @ params[1]['w'] + params[1]['b']

 # # jdbg.print(f"out.dtype {out.dtype} w.dtype {params[2]['w'].dtype} b.dtype {params[2]['b'].dtype}")
 
 # out = out @ params[2]['w'] + params[2]['b']

 for i, d in enumerate(params['ff']):
  out = out @ d['w'] + d['b']
  jdbg.print("ff {i} shape {shape}", i=i, shape=d['w'].shape)
  # if i < len(dense) - 1:
  #  out = jnn.elu(out)
  # else:
  #  out = jnn.sigmoid(out)

 return jnn.sigmoid(out.mean(axis=1))
 # return out.sum(axis=1)
 # return out.mean(axis=1)

# errors = checkify.user_checks | checkify.index_checks | checkify.float_checks

# log2, but adds a very small number to avoid division by zero
# Assumes this doesn't produce more zeros!
# def log2_safe(x):
#  eps = jnp.full(x.shape, 1e-12, dtype=fX)
#  return jnp.log2(x + eps)

# def log2_safe_base(x):
#  jnp.log2(x)

# log2_safe = jax.vmap(log2_safe_base)

def log2_safe(x):
 log2 = jnp.log2(x)
 return jnp.nan_to_num(log2, neginf=0.0) 

def binary_cross_entropy(preds, y):
 h = y * log2_safe(preds) + (1.0 - y) * log2_safe(1.0 - preds)

 return -h.mean()

def mean_squared_error(preds, y):
 delta = preds - y
 return jnp.mean(delta**2, dtype=fX)



@jax.jit
def loss_core(params, x, y):
 preds = model(params, x)
 # jdbg.breakpoint()
 # jdbg.print(f"preds.shape {preds.shape} y.shape {y.shape}")
 # jdbg.print(f"preds.dtype {preds.dtype} y.dtype {y.dtype}")
 # checkify.check(preds.dtype == y.dtype, "preds dtype not equal to labels dtype")
 # checkify.check(preds.shape == y.shape, "predictions and labels had different shapes")
 # delta = preds - y
 # return jnp.mean(delta**2, dtype=fX)
 # return binary_cross_entropy(preds, y)
 return mean_squared_error(preds, y)


# loss = checkify.checkify(loss_core, errors)

# dloss_core = jax.grad(loss_core)
# dloss = checkify.checkify(dloss_core, errors)

loss = loss_core
dloss = jax.grad(loss_core)

@jax.jit
def update(params, x, y, lr=1e-2):
 # pred = model(params, x)

 grad = dloss(params, x, y)

 return jax.tree_map(
     lambda p, g: p - lr * g, params, grad
 )

def fit(params: optax.Params, optimizer: optax.GradientTransformation, x, y, epochs) -> optax.Params:
  opt_state = optimizer.init(params)

  @jax.jit
  def step(params, opt_state, batch, labels):
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

  x_shape_batched = (1, *x.shape)
  y_shape_batched = (1, *y.shape)

  for i in range(epochs):
   for j, (batch, labels) in enumerate(zip(x.reshape(x_shape_batched), y.reshape(y_shape_batched))):
    params, opt_state, loss_value = step(params, opt_state, batch, labels)
   # if i % 100 == 0:
   print(f'step {i}, loss: {loss_value}')

  return params


if __name__ == "__main__":
 data = load(True)
 
 x_train = jnp.array(data['x_train'], dtype=iX)
 y_train = jnp.array(data['y_train'], dtype=fX)
 x_val = jnp.array(data['x_val'], dtype=iX)
 y_val = jnp.array(data['y_val'], dtype=fX)
 x_test = jnp.array(data['x_test'], dtype=iX)
 y_test = jnp.array(data['y_test'], dtype=fX)
 vocab_len = data['vocab_len']

 del data

 print(f"x_train shape: {x_train.shape}")
 print(f"y_train shape: {y_train.shape}")
 print(f"x_val shape: {x_val.shape}")
 print(f"y_val shape: {y_val.shape}")
 print(f"x_test shape: {x_test.shape}")
 print(f"y_test shape: {y_test.shape}")

 print(x_train[0][0])

 # print(x_train[:3])
 # print(y_train[:3])


 rng_key = jrand.PRNGKey(85439357)
 emb_key, attn_key, attn_stack_key, dense0_w_key, dense0_b_key, dense1_w_key, dense1_b_key = jrand.split(rng_key, 7)

 initializer = jnn.initializers.glorot_uniform()

 ATTN_STACK = ATTN_DIMS * ATTN_HEADS

 params = {
  'emb': initializer(emb_key, (vocab_len, EMBEDDING_DIMS), dtype=fX),
  'self-attn': [initializer(attn_key, (EMBEDDING_DIMS, ATTN_DIMS), dtype=fX) for _ in range(ATTN_HEADS)], 
  'self-attn-stack': jrand.uniform(attn_stack_key, (ATTN_STACK,), dtype=fX),
  'ff': [
   {
    'w': initializer(dense0_w_key, (ATTN_STACK , ATTN_STACK // 2), dtype=fX),
    'b': jrand.normal(dense0_b_key, (ATTN_STACK // 2,), dtype=fX)
   },
   {
    'w': initializer(dense1_w_key, (ATTN_STACK // 2, 1), dtype=fX),
    'b': jrand.normal(dense1_b_key, (1,), dtype=fX)
   }
  ]
 }

 train_loss = loss(params, x_train, y_train)
 val_loss = loss(params, x_val, y_val)
 val_preds = model(params, x_val).round()
 val_acc = accuracy(val_preds, y_val)
 print(f"-1 train_loss: {train_loss} val_loss: {val_loss} val_acc: {val_acc}")

 optimizer = optax.adam(learning_rate=1e-3)

 start = time.time()
 params = fit(params, optimizer, x_train, y_train, 200)

 dur = time.time() - start

 print(f"duration: {dur}")


 print(f"y_test shape: {y_test.shape}")

 preds = model(params, x_test).round()

 acc = accuracy(preds, y_test)

 print(f"accuracy: {acc}")

