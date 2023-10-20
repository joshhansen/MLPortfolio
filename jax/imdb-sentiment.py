from typing import Mapping

from collections import Counter
import itertools
import json
import os
import time

import jax
from jax import nn as jnn
from jax import numpy as jnp
from jax import random as jrand
from jax import tree_util as jtree

import matplotlib
import matplotlib.pyplot as plt

from more_itertools import unzip

import numpy as np

#TODO Drop truncation / padding?

LinearParams = Mapping[str, jnp.ndarray]

EMBEDDING_DIMS = 20
fX = jnp.float32
iX = jnp.uint32
# device = torch.device("cuda")

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

  

# class ObjectDataset(Dataset):
#  def __init__(self, obj):
#   self.obj = obj

#  def __getitem__(self, i):
#   return self.obj[i]

#  def __len__(self):
#   return len(self.obj)

# class ImdbSentiment(LightningModule):
#  def __init__(self, vocab_size):
#   super().__init__()
#   self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIMS, dtype=fX)
#   self.linear1 = nn.Linear(EMBEDDING_DIMS, EMBEDDING_DIMS // 2, dtype=fX)
#   self.linear2 = nn.Linear(EMBEDDING_DIMS // 2, 1, dtype=fX)

#   self.acc = Accuracy(task='binary').to(device)
#   self.auroc = AUROC(task='binary').to(device)
#   self.f1score = F1Score(task='binary').to(device)

#   self.metrics = [
#    self.acc, self.auroc, self.f1score
#   ]

#  def params(self):
#   return itertools.chain(self.embedding.parameters(), self.linear1.parameters(), self.linear2.parameters())

#  def forward(self, x, y):
#   emb = self.embedding(x)
#   doc_emb = emb.sum(1)# sum over the word indices
#   out1 = self.linear1(doc_emb)
#   out2 = self.linear2(out1)

#   # print(f"emb {emb.shape} {emb.dtype}")
#   # print(f"doc_emb {doc_emb.shape} {doc_emb.dtype}")
#   # print(f"out1 {out1.shape} {out1.dtype}")
#   # print(f"out2 {out2.shape} {out2.dtype}")

#   return torch.nn.functional.sigmoid(out2)

#  def training_step(self, batch, batch_idx):
#   inputs, target = batch
#   # output = self(inputs, target)
#   output = self.forward(inputs, target)
#   # print(f"out shape: {output.shape} {output.dtype}")
#   # print(f"target shape: {target.shape} {target.dtype}")

#   target_resized = target.view(output.shape)
#   loss = torch.nn.functional.binary_cross_entropy(output, target_resized)

#   self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)

#   # Make specific predictions
#   preds = output.round()

#   for m in self.metrics:
#    mval = m(preds, target_resized)
#    self.log(f"test_{m.__class__.__name__}", mval, on_step=True, on_epoch=True, prog_bar=True)

#   return loss

#  def validation_step(self, batch, batch_idx):
#   inputs, target = batch
#   output = self.forward(inputs, target)

#   target_resized = target.view(output.shape)
#   loss = torch.nn.functional.binary_cross_entropy(output, target_resized)

#   self.log('test_loss', loss, prog_bar=True)

#   # Make specific predictions
#   preds = output.round()

#   for m in self.metrics:
#    mval = m(preds, target_resized)
#    self.log(f"test_{m.__class__.__name__}", mval, prog_bar=True)

#   return loss

#  def test_step(self, batch, batch_idx):
#   inputs, target = batch
#   output = self.forward(inputs, target)

#   target_resized = target.view(output.shape)
#   loss = torch.nn.functional.binary_cross_entropy(output, target_resized)

#   # Make specific predictions
#   preds = output.round()

#   self.log('test_loss', loss, prog_bar=True)

#   for m in self.metrics:
#    mval = m(preds, target_resized)
#    self.log(f"test_{m.__class__.__name__}", mval, prog_bar=True)

#   return loss
  

#  def configure_optimizers(self):
#     return torch.optim.Adam(self.params(), lr=1e-3)

# @jtree.register_pytree_node_class
# class Dense:
#  def __init__(self, rng_key, in_dims, out_dims, dtype=fX):
#   self.rng_key, w_key, b_key = jrand.split(rng_key, 3)
#   self.weights = jrand.normal(w_key, (in_dims, out_dims), dtype=dtype)
#   self.biases = jrand.normal(b_key, (out_dims,), dtype=dtype)

#  def __call__(self, x):
#   return x @ self.weights + self.biases

#  def tree_flatten(self):
#   aux = {
#    'rng_key': self.rng_key,
#    'in_dims': self.weights.shape[0],
#    'out_dims': self.weights.shape[1],
#    'dtype': self.weights.dtype,
#   }
#   return ((self.weights, self.biases), aux)

#  @classmethod
#  def tree_unflatten(cls, aux, children):
#   rng_key = aux['rng_key']
#   in_dims = aux['in_dims']
#   out_dims = aux['out_dims']
#   dtype = aux['dtype']
#   instance = cls(rng_key, in_dims, out_dims, dtype)

#   instance.weights, instance.biases = children

#   return instance


def model(params, x):
 emb, *dense = params

 out = emb[x].sum(axis=1)
 for i, d in enumerate(dense):
  out = out @ d['w'] + d['b']
  if i < len(dense) - 1:
   out = jnn.relu(out)
  else:
   out = jnn.sigmoid(out)

 return out.sum(axis=1)

@jax.jit
def loss(params, x, y):
 preds = model(params, x)
 delta = preds - y
 return jnp.mean(delta**2)

dloss = jax.grad(loss)

@jax.jit
def update(params, x, y, lr=1e-1):
 # pred = model(params, x)

 grad = dloss(params, x, y)

 return jax.tree_map(
     lambda p, g: p - lr * g, params, grad
 )

if __name__ == "__main__":
 path = os.environ['HOME'] + "/Data/com/github/nas5w/imdb-data/reviews.json"

 with open(path) as r:
  raw = json.load(r)

 
 vocab = set()
 vocab.add("__padding__")
 for datum in raw:
  words = datum['t'].split()
  vocab.update(words)

 # print(vocab)
 vocab_len = len(vocab)
 print(f"vocab_len: {vocab_len}")

 word_to_idx = {word: i for i, word in enumerate(vocab)}
 padding_idx = word_to_idx["__padding__"]

 del vocab

 indexed: list[tuple[ list[int], int]] = list()
 lens: Counter = Counter()

 for datum in raw:
  words = datum['t'].split()
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

 x_train = jnp.array(list(x_train_raw))
 del x_train_raw
 
 y_train = jnp.array(list(y_train_raw))
 del y_train_raw

 x_val = jnp.array(list(x_val_raw))
 del x_val_raw

 y_val = jnp.array(list(y_val_raw))
 del y_val_raw

 x_test = jnp.array(list(x_test_raw))
 del x_test_raw
 
 y_test = jnp.array(list(y_test_raw))
 del y_test_raw

 print(f"x_train shape: {x_train.shape}")
 print(f"y_train shape: {y_train.shape}")

 print(x_train[:3])
 print(y_train[:3])


 rng_key = jrand.PRNGKey(85439357)
 emb_key, dense0_w_key, dense0_b_key, dense1_w_key, dense1_b_key = jrand.split(rng_key, 5)

 params = [
  jrand.normal(rng_key, (vocab_len, EMBEDDING_DIMS,)),
  {
   'w': jrand.normal(dense0_w_key, (EMBEDDING_DIMS, EMBEDDING_DIMS // 2), dtype=fX),
   'b': jrand.normal(dense0_b_key, (EMBEDDING_DIMS // 2,), dtype=fX)
  },
  {
   'w': jrand.normal(dense1_w_key, (EMBEDDING_DIMS // 2, 1), dtype=fX),
   'b': jrand.normal(dense1_b_key, (1,))
  }
 ]

 start = time.time()
 for i in range(1000):
  print(f"\r{i}     ")
  params = update(params, x_train, y_train)

  val_preds = model(params, x_val)

  val_grads = dloss(params, x_val, y_val)

  print(f"val preds shape: {val_preds.shape}")

  print(f"val grads: {val_grads}")
  

  print(f"{i} loss: {loss(params, x_train, y_train)}")

 dur = time.time() - start

 print(f"duration: {dur}")

 print(f"y_test shape: {y_test.shape}")

 preds = model(params, x_test)

 print(f"preds shape: {preds.shape}")

 matching = y_test == preds

 print(f"matching shape: {matching.shape}")

 correct = matching.sum()

 print(f"# correct: {correct}")

 print(f"correct shape: {correct.shape}")

 accuracy = correct / x_test.shape[0]

 print(f"accuracy: {accuracy}")

 # matplotlib.use('qtagg')
 # plt.scatter(model(params, x_train), y_train)
 # plt.show()

 # w, b = params
 # print(f"w: {w:<.2f}, b: {b:<.2f}")

 # print(dir(train))

 # train_loader = DataLoader(train, shuffle=True, batch_size=100)
 # val_loader = DataLoader(val, batch_size=100)
 # test_loader = DataLoader(test, batch_size=100)

 # model = ImdbSentiment(len(vocab))

 # trainer = Trainer(max_epochs=50)
 # trainer.fit(model, train_loader, val_loader)

 # trainer.test(ckpt_path='best', dataloaders=test_loader)
