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
# from jax import debug as jdbg
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrand
from jax import tree_util as jtree

from jax.experimental import checkify

import matplotlib
import matplotlib.pyplot as plt

from more_itertools import unzip

import numpy as np

import optax

EMBEDDING_DIMS = 20
# EMBEDDING_DIMS = 2
fX = jnp.float32
iX = jnp.uint32

np.random.seed(48349834)

np_rng = np.random.default_rng()
def random_split(data, weights):
 parts = [ list() for _ in weights ]

 for datum in data:
  x = np_rng.random()

  total = 0.0
  for i, weight in enumerate(weights):
   total += weight

   if x <= total:
    parts[i].append(datum)
    break

 return parts

def accuracy(preds, y):
 matching = y == preds

 correct = matching.sum()

 return correct / x_test.shape[0]

def model(params, x):
 emb, *dense = params

 # out = emb[x].mean(axis=1)
 out = emb[x].sum(axis=1)

 # # jdbg.print(f"out.dtype {out.dtype} w.dtype {params[1]['w'].dtype} b.dtype {params[1]['b'].dtype}")

 # out = out @ params[1]['w'] + params[1]['b']

 # # jdbg.print(f"out.dtype {out.dtype} w.dtype {params[2]['w'].dtype} b.dtype {params[2]['b'].dtype}")
 
 # out = out @ params[2]['w'] + params[2]['b']

 for i, d in enumerate(dense):
  out = out @ d['w'] + d['b']
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

token_rgx = re.compile("(?:[A-Za-z0-9]+'[A-Za-z0-9]+)|[A-Za-z0-9]+")
def tokenize(s):
 return [s.lower() for s in token_rgx.findall(s)]


def fit(params: optax.Params, optimizer: optax.GradientTransformation, x, y) -> optax.Params:
  opt_state = optimizer.init(params)

  @jax.jit
  def step(params, opt_state, batch, labels):
    loss_value, grads = jax.value_and_grad(loss)(params, batch, labels)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_value

  x_shape_batched = (1, *x.shape)
  y_shape_batched = (1, *y.shape)

  for i in range(100):
   for j, (batch, labels) in enumerate(zip(x.reshape(x_shape_batched), y.reshape(y_shape_batched))):
    params, opt_state, loss_value = step(params, opt_state, batch, labels)
   # if i % 100 == 0:
   print(f'step {i}, loss: {loss_value}')

  return params


if __name__ == "__main__":
 path = os.environ['HOME'] + "/Data/com/github/nas5w/imdb-data/reviews.json"

 with open(path) as r:
  raw = json.load(r)

 
 vocab = set()
 vocab.add("__padding__")
 for datum in raw:
  words = tokenize(datum['t'])
  vocab.update(words)

 vocab = list(vocab)
 vocab.sort()

 # print(vocab)
 vocab_len = len(vocab)
 print(f"vocab_len: {vocab_len}")

 word_to_idx = {word: i for i, word in enumerate(vocab)}
 padding_idx = word_to_idx["__padding__"]

 del vocab

 indexed: list[tuple[ list[int], int]] = list()
 lens: Counter = Counter()

 for datum in raw:
  words = tokenize(datum['t'])
  word_indices = [ word_to_idx[word] for word in words ]

  lens[len(word_indices)] += 1

  class_ = datum['s']
  indexed.append((word_indices, class_))
 
 del raw
 del word_to_idx

 sorted_lens: list[tuple[int,int]] = list(lens.items())
 sorted_lens.sort(key = lambda x: x[0])
 total_lens = lens.total()
 del lens

 cum = 0
 target_len = -1
 for l, n in sorted_lens:
  cum += n
  pct = cum / total_lens
  if pct >= 0.95:
   target_len = l
   break

 target_len = 50

 print(f"target_len: {target_len}")
 print(f"padding_idx: {padding_idx}")

 del sorted_lens

 data = list()
 for x, y in indexed:
  # Pad to target_len
  if len(x) < target_len:
   x.extend([padding_idx] * (target_len - len(x)))
  else:
   x = x[:target_len]

  data.append((jnp.array(x, dtype=iX), jnp.array(y, dtype=fX)))

 del indexed
 
 print(f"data: {len(data)}")

 train, val, test = random_split(data, [0.8, 0.1, 0.1])
 del data

 print(f"train: {len(train)}")
 print(f"val: {len(val)}")
 print(f"test: {len(test)}")

 x_train_raw, y_train_raw = unzip(train)
 x_val_raw, y_val_raw = unzip(val)
 x_test_raw, y_test_raw = unzip(val)

 x_train = jnp.array(list(x_train_raw), dtype=iX)
 del x_train_raw
 
 y_train = jnp.array(list(y_train_raw), dtype=fX)
 del y_train_raw

 x_val = jnp.array(list(x_val_raw), dtype=iX)
 del x_val_raw

 y_val = jnp.array(list(y_val_raw), dtype=fX)
 del y_val_raw

 x_test = jnp.array(list(x_test_raw), dtype=iX)
 del x_test_raw
 
 y_test = jnp.array(list(y_test_raw), dtype=fX)
 del y_test_raw

 print(f"x_train shape: {x_train.shape}")
 print(f"y_train shape: {y_train.shape}")

 print(x_train[0][0])

 # print(x_train[:3])
 # print(y_train[:3])


 rng_key = jrand.PRNGKey(85439357)
 emb_key, dense0_w_key, dense0_b_key, dense1_w_key, dense1_b_key = jrand.split(rng_key, 5)

 initializer = jnn.initializers.glorot_uniform()

 params = [
  # jrand.normal(emb_key, (vocab_len, EMBEDDING_DIMS,), dtype=fX),
  initializer(emb_key, (vocab_len, EMBEDDING_DIMS), dtype=fX),
  {
   # 'w': jrand.normal(dense0_w_key, (EMBEDDING_DIMS, EMBEDDING_DIMS // 2), dtype=fX),
   # 'w': initijnn.initializers.glorot_uniform(EMBEDDING_DIMS, EMBEDDING_DIMS // 2, dtype=fX),
   'w': initializer(dense0_w_key, (EMBEDDING_DIMS, EMBEDDING_DIMS // 2), dtype=fX),
   'b': jrand.normal(dense0_b_key, (EMBEDDING_DIMS // 2,), dtype=fX)
  },
  {
   # 'w': jrand.normal(dense1_w_key, (EMBEDDING_DIMS // 2, 1), dtype=fX),
   # 'w': jnn.initializers.glorot_uniform(EMBEDDING_DIMS // 2, 1, dtype=fX),
   'w': initializer(dense1_w_key, (EMBEDDING_DIMS // 2, 1), dtype=fX),
   'b': jrand.normal(dense1_b_key, (1,), dtype=fX)
  }
 ]

 train_loss = loss(params, x_train, y_train)
 val_loss = loss(params, x_val, y_val)
 val_preds = model(params, x_val).round()
 val_acc = accuracy(val_preds, y_val)
 print(f"-1 train_loss: {train_loss} val_loss: {val_loss} val_acc: {val_acc}")

 best_params = params
 best_loss = val_loss

 optimizer = optax.adam(learning_rate=1e-2)

 start = time.time()
 params = fit(params, optimizer, x_train, y_train)
 # for i in range(10000):
 #  params = update(params, x_train, y_train)

 #  if i % 10 == 0:
 #   train_loss = loss(params, x_train, y_train)
 #   val_loss = loss(params, x_val, y_val)
 #   val_preds = model(params, x_val).round()
 #   val_acc = accuracy(val_preds, y_val)

 #   print(f"{i} train_loss: {train_loss} val_loss: {val_loss} val_acc: {val_acc}")

 #   if val_loss < best_loss:
 #    best_params = params
 #    best_loss = val_loss 

 dur = time.time() - start

 print(f"duration: {dur}")


 print(f"y_test shape: {y_test.shape}")

 preds = model(params, x_test).round()

 acc = accuracy(preds, y_test)

 print(f"accuracy: {acc}")

