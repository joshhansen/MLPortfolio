# Infer Bayes-rule-respecting knowledge from text
from collections import defaultdict
from itertools import pairwise
import os
import re
from statistics import fmean
from typing import Generator


import nltk
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

def sentences(path: str) -> Generator[str, None, None]:
 with open(path) as r:
  text = r.read()

 text = text.replace('\n', ' ')

 return nltk.tokenize.sent_tokenize(text)
 
class Index:
 def __init__(self):
  self._next = 0
  self._idx = dict()
  self._unidx = dict()

 def idx(self, s: str) -> int:
  try:
   return self._idx[s]
  except:
   n = self._next
   self._idx[s] = n
   self._next += 1
   return n

 def idx_tokens(self, tokens: list[str]) -> list[int]:
  indices: list[int] = list()

  for t in tokens:
   i = self.idx(t)
   indices.append(i)

  return indices

 def unidx(self, i: int) -> str:
  return self._unidx[i]

 def __len__(self) -> int:
  return self._next
  
   
def tokenize_words(s: str, max_len: int, pad='<pad>') -> list[str]:
 words: list[str] = list()

 for m in word_rgx.finditer(s):
  if len(words) < max_len:
   words.append(m.group(0))

 if len(words) < max_len:
  padding_needed = max_len - len(words)
  padding = [pad] * padding_needed
  words.extend(padding)

 return words
  

def _gen_sentences_dataset(path: str, vocab: Index, seq_len: int):
 print(f"Sentences initializing for {path}")
 for s in sentences(poe_path):
  s = s.lower()

  tokens = tokenize_words(s, MAX_SEQ_LEN)

  indices = vocab.idx_tokens(tokens)

  yield tf.constant(indices, dtype=tf.int32)


def sentences_dataset(path: str, vocab: Index, seq_len: int):
    gen = lambda: _gen_sentences_dataset(path, vocab, seq_len)

    return tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(seq_len,), dtype=tf.int32))

def positional_encoding(length: int, depth: float):
  depth = depth/2

  positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

  angle_rates = 1 / (10000**depths)         # (1, depth)
  angle_rads = positions * angle_rates      # (pos, depth)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
 def __init__(self, *, vocab_size: int, emb: int):
  super().__init__()
  self.emb_size = emb
  self.embedding = tf.keras.layers.Embedding(vocab_size, emb, mask_zero=True) 
  self.pos_encoding = positional_encoding(length=2048, depth=emb)

 def compute_mask(self, *args, **kwargs):
  return self.embedding.compute_mask(*args, **kwargs)

 def call(self, x: tf.Tensor):
  # print(f"x.get_shape: {x.get_shape()}")
  length = x.get_shape()[1]
  # print(f"x.get_shape()[1]: {x.get_shape()[1]}")
  x = self.embedding(x)
  # This factor sets the relative scale of the embedding and positonal_encoding.
  x *= tf.math.sqrt(tf.cast(self.emb_size, tf.float32))
  x = x + self.pos_encoding[tf.newaxis, :length, :]
  return x

class MhaResLayerNorm(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int):
  super().__init__()
  self.mha = tf.keras.layers.MultiHeadAttention(
   num_heads=num_heads,
   key_dim=emb,
  )
  self.layernorm = tf.keras.layers.LayerNormalization()
  self.add = tf.keras.layers.Add()

 def call(self, x: tf.Tensor):
  # (batch, seq, emb)
  attn = self.mha(
   query=x,
   value=x,
   key=x
  )
  # (batch, seq, emb)

  x = self.add([x, attn])
  # (batch, seq, emb)

  x = self.layernorm(x)
  # (batch, seq, emb)

  return x

class EncodingSummer(tf.keras.layers.Layer):
 def call(self, x: tf.Tensor):
  # (batch, seq, emb)
  x = tf.reduce_sum(x, 1)
  # (batch, emb)
  return x

# Encodes a text sequence as "knowledge" meant to be subject to Bayes' rule,
# plus a vector not subject to Bayes' rule.
class Encoder(tf.keras.layers.Layer):
 # The "knowledge" dimension is the same as the emb dimension because it's easier that way in TF's MHA implementation
 # The number of "propositions" is the same as the sequence length for the same reason
 # A proper seq-to-seq model could allow those to differ
 def __init__(self, *, num_heads: int, input_vocab_size: int, emb: int, non_knowledge_dim: int):
  super().__init__()
  self.pos = PositionalEmbedding(
   vocab_size=input_vocab_size,
   emb=emb,
  )
  self.knowledge_extractor = MhaResLayerNorm(num_heads=num_heads, emb=emb)
  self.non_knowledge_extractor = tf.keras.Sequential([
   MhaResLayerNorm(num_heads=num_heads, emb=emb),
   EncodingSummer(),
   tf.keras.layers.Dense(non_knowledge_dim)
  ])

 def call(self, x: tf.Tensor):
  # (batch, seq)

  x = self.pos(x)

  # (batch, seq, emb)

  k = self.knowledge_extractor(x)
  # (batch, seq, emb)
  
  j = self.non_knowledge_extractor(x)
  # (batch, non_knowledge_dim)

  return (k, j)

# "Knowledge" and non-knowledge content to text
class Decoder(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int, output_vocab_size: int, seq_len: int, non_knowledge_dim: int):
  super().__init__()
  self.pos_encoding = positional_encoding(length=seq_len, depth=emb+non_knowledge_dim)
  self.dec = MhaResLayerNorm(num_heads=num_heads, emb=emb+non_knowledge_dim)
  self.emb_size = emb
  self.emb_to_vocab = tf.keras.layers.Dense(output_vocab_size)
  self.softmax = tf.keras.layers.Softmax()
  self.seq_len = seq_len

 def call(self, *, k: tf.Tensor, j: tf.Tensor):
  # k: (batch, seq, emb)
  # j: (batch, non_knowledge_dim)

  j = tf.expand_dims(j, 1)

  # j: (batch, 1, non_knowledge_dim)

  s = list(j.shape)
  s[1] = self.seq_len
  j = tf.broadcast_to(j, s)

  # j: (batch, seq, non_knowledge_dim)

  x = tf.concat([k, j], -1)
  # (batch, seq, emb+non_knowledge_dim)

  # Augment with a position encoding
  x = x + self.pos_encoding[tf.newaxis, :self.seq_len, :]
  # (batch, seq, emb+non_knowledge_dim)

  x = self.dec(x)
  # (batch, seq, emb+non_knowledge_dim)

  x = self.emb_to_vocab(x)
  # (batch, seq, token)

  x = self.softmax(x)
  # (batch, seq, token)

  return x

# P(x) in bulk
class MarginalDist(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int):
  super().__init__()
  self.attn = MhaResLayerNorm(num_heads=num_heads, emb=emb)
  self.ff = tf.keras.layers.Dense(1)
  self.normal_pdf = tfp.distributions.Normal(0, 1)

 def call(self, x: tf.Tensor):
  # (batch, seq, emb)

  x = self.attn(x)

  # (batch, seq, emb)

  x = self.ff(x)

  # (batch, seq, 1)

  x = self.normal_pdf.prob(x)

  # (batch, seq, 1), dtype float [0,+inf]

  x = tf.squeeze(x, 2)

  # (batch, seq), dtype float [0,+inf]

  return x

 
# P(y|x) in bulk
#
# NOTE Could this hand off to MarginalDist ?
class ConditionalDist(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int):
  super().__init__()
  self.attn = MhaResLayerNorm(num_heads=num_heads, emb=2*emb)
  self.ff = tf.keras.layers.Dense(1)
  self.normal_pdf = tfp.distributions.Normal(0, 1)

 def call(self, x: tf.Tensor, y: tf.Tensor):
  # (batch, seq, emb)^2

  z = tf.concat([x, y], -1)

  # (batch, seq, 2*emb)

  z = self.attn(z)

  # (batch, seq, 2*emb)

  z = self.ff(z)

  # (batch, seq, 1)

  z = self.normal_pdf.prob(z)

  # (batch, seq, 1), dtype float [0,+inf]

  z = tf.squeeze(z, 2)

  # (batch, seq), dtype float [0,+inf]

  return z

# class EncDec(tf.keras.Model):
#  def __init__(self, *, num_heads: int, emb:int, vocab_size: int):
#   super().__init__()
#   self.enc = Encoder(
#    num_heads=num_heads,
#    input_vocab_size=vocab_size,
#    emb=emb,
#   )
#   self.dec = Decoder(
#    num_heads=num_heads,
#    emb=emb,
#    output_vocab_size=vocab_size,
#   )

#  # def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
#   # x, y = inputs
#  def call(self, x: tf.Tensor):
#   x_enc = self.enc(x)
#   return self.dec(x_enc)

 
# Autoencoder training step
#
# x shape: (batch, seq)
# @tf.function
def enc_dec_train_step(*,
 enc_opt: tf.keras.optimizers.Optimizer,
 dec_opt: tf.keras.optimizers.Optimizer,
 enc: Encoder,
 dec: Decoder,
 x: tf.Tensor,
 loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(),
 train = True,
):
 with tf.GradientTape(persistent=True) as tape:
  k, j = enc(x)
  y = dec(k=k, j=j)
  loss = loss_fn(x, y)

 if train:
  enc_grads = tape.gradient(loss, enc.trainable_weights)
  dec_grads = tape.gradient(loss, dec.trainable_weights)
  enc_opt.apply_gradients(zip(enc_grads, enc.trainable_weights))
  dec_opt.apply_gradients(zip(dec_grads, dec.trainable_weights))

 return loss.numpy()

# Bayes' rule training step
#
# x shape: (batch, seq)
# y shape: (batch, seq)
# @tf.function
def bayes_train_step(*,
 enc_opt: tf.keras.optimizers.Optimizer,
 f_opt: tf.keras.optimizers.Optimizer,
 h_opt: tf.keras.optimizers.Optimizer,
 enc: Encoder,
 f: MarginalDist,
 h: ConditionalDist,
 x: tf.Tensor,
 y: tf.Tensor,
 train = True,
):
 with tf.GradientTape(persistent=True) as tape:
  k_x, _j_x = enc(x)
  k_y, _j_y = enc(y)

  # How badly do the probability distributions respect Bayes' rule?
  loss = ( h(k_x, k_y) - h(k_y, k_x) * f(k_x) / f(k_y) )**2 
 
 if train:
  enc_grads = tape.gradient(loss, enc.trainable_weights)
  enc_opt.apply_gradients(zip(enc_grads, enc.trainable_weights))
  f_grads = tape.gradient(loss, f.trainable_weights)
  f_opt.apply_gradients(zip(f_grads, f.trainable_weights))
  h_grads = tape.gradient(loss, enc.trainable_weights)
  h_opt.apply_gradients(zip(h_grads, enc.trainable_weights))

 return loss.numpy()

def train(*,
 enc_opt: tf.keras.optimizers.Optimizer,
 dec_opt: tf.keras.optimizers.Optimizer,
 f_opt: tf.keras.optimizers.Optimizer,
 h_opt: tf.keras.optimizers.Optimizer,
 epochs: int,
 steps_per_epoch: int,
 valid_steps_per_epoch: int,
 enc: Encoder,
 dec: Decoder,
 f: MarginalDist,
 h: ConditionalDist,
 train: tf.data.Dataset,# infinitely looped, shuffled
 valid: tf.data.Dataset,# infinitely looped, shuffled
):
 train_iter = iter(train)
 valid_iter = iter(valid)
 train_pairs = pairwise(train_iter)
 valid_pairs = pairwise(valid_iter)
 
 #TODO? bayes training from fully random (synthetic) pairs

 for e in range(epochs):
  print(f"Epoch {e}/{epochs}")

  losses = defaultdict(list)
  
  for s in range(steps_per_epoch):
   a,b = next(train_pairs)

   print(a.shape)
   print(b.shape)
   
   # autoencoder roundtrips
   for x in (a,b):
    losses['enc_dec'].append(enc_dec_train_step(
     enc_opt=enc_opt,
     dec_opt=dec_opt,
     enc=enc,
     dec=dec,
     x=x,
    ))

   # bayes rule error
   losses['bayes_rule'].append(bayes_train_step(
     enc_opt=enc_opt,
     f_opt=f_opt,
     h_opt=h_opt,
     enc=enc,
     f=f,
     h=h,
     x=a,
     y=b,
   ))

   loss_keys = list(sorted(losses.keys()))
   for k in loss_keys:
    l = fmean(losses[k])
    
    print(f"{k}: {l}")

  valid_losses = defaultdict(list)

  for s in range(valid_steps_per_epoch):
   a,b = next(valid_pairs)

   # autoencoder roundtrips
   for x in (a,b):
    valid_losses['enc_dec'].append(enc_dec_train_step(
     enc_opt=enc_opt,
     dec_opt=dec_opt,
     enc=enc,
     dec=dec,
     x=x,
     train=False,
    ))

   # bayes rule error
   valid_losses['bayes_rule'].append(bayes_train_step(
     enc_opt=enc_opt,
     f_opt=f_opt,
     h_opt=h_opt,
     enc=enc,
     f=f,
     h=h,
     x=a,
     y=b,
     train=False,
   ))
  
   valid_loss_keys = list(sorted(valid_losses.keys()))
   for k in valid_loss_keys:
    l = fmean(valid_losses[k])
    
    print(f"valid {k}: {l}")

word_rgx = re.compile("[A-Za-z0-9'-]+")
if __name__ == "__main__":
 home_dir = os.environ['HOME']
 data_dir = os.path.join(home_dir, 'Data')
 guten_dir = os.path.join(data_dir, 'org', 'gutenberg')

 poe_path = os.path.join(guten_dir, 'pg10031-poe-no-pg.txt')

 MAX_SEQ_LEN = 64
 BATCH=10

 vocab = Index()
 sentence_vecs: list[list[int]] = list()
 # word_tokenizer = tft.UnicodeScriptTokenizer()

 for _ in sentences_dataset(poe_path, vocab, MAX_SEQ_LEN):
  pass# Do this to full populate the vocabulary

 def dataset(filter):
  return sentences_dataset(poe_path, vocab, MAX_SEQ_LEN)\
   .repeat()\
   .shuffle(1000)\
   .enumerate()\
   .filter(filter)\
   .map(lambda i, x: x)\
   .batch(BATCH)
  
 txt_train = dataset(lambda i, x: i % 10 > 1)
 # txt_test = dataset(lambda i, x: i == 0)
 txt_valid = dataset(lambda i, x: i == 1)

 num_heads = 8
 emb = 32
 non_knowledge_dim = 16

 enc_opt = Adam()
 dec_opt = Adam()
 f_opt = Adam()
 h_opt = Adam()

 enc = Encoder(
  num_heads=num_heads,
  input_vocab_size=len(vocab),
  emb=emb,
  non_knowledge_dim=non_knowledge_dim,
 )
 dec = Decoder(
  num_heads=num_heads,
  output_vocab_size=len(vocab),
  emb=emb,
  non_knowledge_dim=non_knowledge_dim,
  seq_len=MAX_SEQ_LEN,
 )
 f = MarginalDist(
  num_heads=num_heads,
  emb=emb,
 )
 h = ConditionalDist(
  num_heads=num_heads,
  emb=emb,
 )

 # emb_dataset_ = emb_dataset(batch=BATCH, emb=EMB)

 train(
  enc_opt=enc_opt,
  dec_opt=dec_opt,
  f_opt=f_opt,
  h_opt=h_opt,
  epochs=20,
  steps_per_epoch=1000,
  valid_steps_per_epoch=100,
  enc=enc,
  dec=dec,
  f=f,
  h=h,
  train=txt_train,
  valid=txt_valid,
 )
