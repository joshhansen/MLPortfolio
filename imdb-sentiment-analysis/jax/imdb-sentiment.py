from imdb_sa_common import load

from typing import Generator, Iterable, Mapping, Tuple

from collections import Counter
from dataclasses import dataclass
import itertools
import json
import os
import re
import time


from jax import config
config.update("jax_debug_nans", True)
config.update("jax_numpy_rank_promotion", "raise")

import jax
from jax import debug as jdbg
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrand
from jax import tree_util as jtree

from jax.experimental import checkify

from more_itertools import unzip

import optax

ITERATIONS = 100
BATCH_SIZE = 1000
EMBEDDING_DIMS = 20
ATTN_QUERIES = 8
ATTN_DIMS = 20
ATTN_HEADS = 5
ATTN_DIMS_PER_HEAD = ATTN_DIMS // ATTN_HEADS
ATTN_SCALE = 1.0 / jnp.sqrt(ATTN_DIMS)
fX = jnp.float32
iX = jnp.uint32

def batches(*arrays) -> Generator[jnp.ndarray, None, None]:
 size = len(arrays[0])
 for arr in arrays[1:]:
  if len(arr) != size:
   raise Exception(f"Size of input {len(arr)} != {size}")

 for start in range(0, size, BATCH_SIZE):
  end = start + BATCH_SIZE

  batches = [arr[start:end] for arr in arrays]

  for batch in batches:
   if len(batch) != BATCH_SIZE:
    # We're at the end of the data where batches are incomplete; get outta here
    return

  yield batches


def scaled_dot_product_attention(q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
 n_q = q.shape[-2]
 d_k = q.shape[-1]

 n_k = k.shape[-2]
 # checkify.check(d_k == k.shape[-1], "q and k d_k mismatch")

 # checkify.check(n_k == v.shape[-2], "k and v n_k mismatch")
 d_v = v.shape[-1]
 
 out = q @ k.transpose()

 out = out / jnp.sqrt(d_k)

 out = jnn.softmax(out)

 return out @ v

@dataclass
class AttentionParams:
 w_query: jnp.ndarray
 w_keys: jnp.ndarray
 w_values: jnp.ndarray

 @classmethod
 def initialize(cls, rng_key, init: jnn.initializers.Initializer, d_model: int, d_k_out: int, d_v_out: int):
  rng_key_q, rng_key_k, rng_key_v = jrand.split(rng_key, 3)
  w_query = init(rng_key_q, (d_model, d_k_out), dtype=fX)
  w_keys = init(rng_key_k, (d_model, d_k_out), dtype=fX)
  w_values = init(rng_key_v, (d_model, d_v_out), dtype=fX)

  return cls(w_query, w_keys, w_values)

@dataclass
class MultiheadAttentionParams:
 heads: list[AttentionParams]
 w: jnp.ndarray

 @classmethod
 def initialize(cls, rng_key, init: jnn.initializers.Initializer, n_heads: int, d_model: int, d_k_out: int, d_v_out: int):
  rng_keys = jrand.split(rng_key, n_heads + 1)

  heads = [ AttentionParams.initialize(rng_keys[i], init, d_model, d_k_out, d_v_out) for i in range(n_heads) ]
  w = init(rng_keys[-1], (n_heads * d_k_out, d_model), dtype=fX)

  return cls(heads, w)

def flatten_AttentionParams(params: AttentionParams) -> Tuple[list[jnp.ndarray], None]:
 return ([params.w_query, params.w_keys, params.w_values], None)

def unflatten_AttentionParams(aux_data: str, flat_contents: list[jnp.ndarray]) -> AttentionParams:
 return AttentionParams(*flat_contents)

jax.tree_util.register_pytree_node(AttentionParams, flatten_AttentionParams, unflatten_AttentionParams)

def flatten_MultiheadAttentionParams(params: MultiheadAttentionParams) -> Tuple[list[jnp.ndarray], None]:
 heads = jnp.array(len(params.heads))

 flat_contents = [params.w, *params.heads]

 return (flat_contents, None)

def unflatten_MultiheadAttentionParams(aux_data: str, flat_contents: list[jnp.ndarray]) -> MultiheadAttentionParams:
 w = flat_contents[0]
 heads = flat_contents[1:]

 return MultiheadAttentionParams(heads, w)
 
jax.tree_util.register_pytree_node(MultiheadAttentionParams, flatten_MultiheadAttentionParams, unflatten_MultiheadAttentionParams)

@dataclass
class Linear:
 w: jnp.ndarray
 b: jnp.ndarray

 @classmethod
 def initialize(cls, key, in_dims: int, out_dims: int, dtype):
  w_key, b_key = jrand.split(key, 2)
  w = initializer(w_key, (in_dims, out_dims), dtype=dtype)
  b = jrand.normal(b_key, (out_dims,), dtype=dtype)
  return cls(w, b)

 def __call__(self, x: jnp.ndarray):
  return x @ self.w + self.b

def flatten_Linear(params: Linear) -> Tuple[list[jnp.ndarray], None]:
 return ([params.w, params.b], None)

def unflatten_Linear(aux_data: str, flat_contents: list[jnp.ndarray]) -> Linear:
 return Linear(*flat_contents)

jax.tree_util.register_pytree_node(Linear, flatten_Linear, unflatten_Linear)

def multihead_attention(params: MultiheadAttentionParams, q: jnp.ndarray, k: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
 attns = list()

 # print(f"q shape: {q.shape}")
 # print(f"k shape: {k.shape}")
 # print(f"v shape: {v.shape}")

 for head in params.heads:
  q_head = q @ head.w_query
  k_head = k @ head.w_keys
  v_head = v @ head.w_values

  attns.append(scaled_dot_product_attention(q_head, k_head, v_head))

 attns_shapes = [ attn.shape for attn in attns ]

 # print(f"attns_shapes: {attns_shapes}")

 out = jnp.concatenate(attns, axis=-1)

 return out @ params.w

batch_multihead_attention = jax.vmap(multihead_attention, [None, None, 0, 0])

def accuracy(preds, y):
 matching = y == preds

 correct = matching.sum()

 return correct / x_test.shape[0]

def model(params, x: jnp.ndarray):
 # print(f"x shape {x.shape}")
 
 # out = emb[x].mean(axis=1)
 out = params['emb'][x]

 # print(f"embedded shape {out.shape}")

 out = batch_multihead_attention(params['attn'], params['attn-query'], out, out)

 # print(f"post-attn shape: {out.shape}")

 out = jnp.mean(out, axis=-2)

 # print(f"post-attn-mean-shape: {out.shape}")

 with jax.numpy_rank_promotion("warn"):
  out = params['linear1'](out)

  # print(f"post-linear1 shape: {out.shape}")

  out = params['linear2'](out)

  # print(f"post-linear2 shape: {out.shape}")

 out = out.mean(axis=-1)

 # print(f"post-dense-mean shape: {out.shape}")

 # print(f"post-mean shape: {out.shape}")

 return jnn.sigmoid(out)
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
def loss(params, x: jnp.ndarray, y: jnp.ndarray):
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

def fit(
 params,
 optimizer: optax.GradientTransformation,
 x_train: jnp.ndarray,
 y_train: jnp.ndarray,
 x_val: jnp.ndarray,
 y_val: jnp.ndarray,
 epochs: int
) -> optax.Params:
 opt_state = optimizer.init(params)

 @jax.jit
 def step(params, opt_state, batch, labels):
  loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
  # print("got grads")
  updates, opt_state = optimizer.update(grads, opt_state, params)
  # print("updated")
  params = optax.apply_updates(params, updates)
  # print("applied")
  return params, opt_state, loss_value

 for i in range(epochs):
  training_losses = list()
  for x_batch, y_batch in batches(x_train, y_train):
   params, opt_state, loss_value = step(params, opt_state, x_batch, y_batch)
   training_losses.append(loss_value)

  training_loss = jnp.mean(jnp.array(training_losses))

  val_losses = list()
  val_preds = list()
  val_labels = list()
  for x_batch, y_batch in batches(x_val, y_val):
   loss_value = loss(params, x_batch, y_batch)
   val_losses.append(loss_value)

   preds = model(params, x_batch).round()
   val_preds.append(preds)
   val_labels.append(y_batch)

  val_preds_all = jnp.concatenate(val_preds)
  val_labels_all = jnp.concatenate(val_labels)
  
  val_loss = jnp.mean(jnp.array(val_losses))
  val_acc = accuracy(val_preds_all, val_labels_all)

  print(f'step {i}, training_loss: {training_loss} val_loss: {val_loss} val_acc: {val_acc}')

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
 target_len = data['target_len']

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
 emb_key, attn_key, attn_query_key, dense0_key, dense1_key = jrand.split(rng_key, 5)

 initializer = jnn.initializers.glorot_uniform()

 params = {
  'emb': initializer(emb_key, (vocab_len, EMBEDDING_DIMS), dtype=fX),
  'attn': MultiheadAttentionParams.initialize(attn_key, initializer, ATTN_HEADS, EMBEDDING_DIMS, EMBEDDING_DIMS, EMBEDDING_DIMS),
  'attn-query': initializer(attn_query_key, (target_len, EMBEDDING_DIMS), dtype=fX),
  'linear1': Linear.initialize(dense0_key, ATTN_DIMS, ATTN_DIMS // 2, fX),
  'linear2': Linear.initialize(dense1_key, ATTN_DIMS // 2, 1, fX),
 }

 total_size = 0
 for layer_name, layer in params.items():
  sizes = jtree.tree_map(lambda x: x.size, layer)
  size = jtree.tree_reduce(lambda x,y: x+y, sizes)
  print(f"{layer_name}: {size}")

  total_size += size

 print(f"total_size: {total_size}")

 shapes = jtree.tree_map(lambda x: x.shape, params)

 print(f"shapes: {shapes}")

 del initializer
 del sizes
 del total_size
 del shapes

 optimizer = optax.adam(learning_rate=1e-3)

 start = time.time()

 params = fit(params, optimizer, x_train, y_train, x_val, y_val, ITERATIONS)

 dur = time.time() - start

 print(f"duration: {dur}")


 print(f"y_test shape: {y_test.shape}")

 preds = list()
 ys = list()

 for x,y in batches(x_test, y_test):
  preds.append(model(params, x).round())
  ys.append(y)

 relevant_preds = jnp.concatenate(preds)
 relevant_ys = jnp.concatenate(ys)

 acc = accuracy(relevant_preds, relevant_ys)

 print(f"accuracy: {acc}")

