import os

import numpy as np
import tensorflow as tf
import tensorflow_text as tft

from grapheme_idx import GraphemeIdx, load_grapheme_idx
from wptitles import wp_titles_dataset

BATCH=10
MAX_STR_LEN = 32

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

def prepare_batch(x: tf.Tensor, y: tf.Tensor):
    x = x[:, :-1]# Drop end tokens

    y = y[:, 1:] # Drop start tokens

    return x, y

# Encodes a sequence as an emb-length vector with no other context
class Enc(tf.keras.layers.Layer):
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
  x = tf.reduce_sum(x, 1)
  # (batch, emb)
  x = self.layernorm(x)
  # (batch, emb)
  return x

class Encoder(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, input_vocab_size: int, emb: int):
  super().__init__()
  self.pos = PositionalEmbedding(
   vocab_size=input_vocab_size,
   emb=emb,
  )
  self.enc = Enc(
   num_heads=num_heads,
   emb=emb
  )

 def call(self, x: tf.Tensor):
  # (batch, seq)

  x = self.pos(x)

  # (batch, seq, emb)

  x = self.enc(x)

  # (batch, emb)

  return x

class Decoder(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int, output_vocab_size: int):
  super().__init__()
  self.mha = tf.keras.layers.MultiHeadAttention(
   num_heads=num_heads,
   key_dim=emb,
  )
  self.emb_to_vocab = tf.keras.layers.Dense(output_vocab_size)

 def call(self, x: tf.Tensor):
  # (batch, emb)

  x = tf.repeat(x, MAX_STR_LEN, 1)

  # (batch, seq, emb)

  x = self.mha(
   query=x,
   value=x,
   key=x,
  )

  # (batch, seq, emb)

  x = self.emb_to_vocab(x)

  # (batch, seq, vocab)

  x = tf.math.argmax(x, 2)

  # (batch, seq) ???

  return x

  

if __name__=="__main__":
  home_dir = os.path.expanduser('~')

  grapheme_idx = load_grapheme_idx()
  print(grapheme_idx)

  wptitles_train_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_train.gz')
  wptitles_valid_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_valid.gz')
  # wptitles_test_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_test.gz')

  tokenizer = tft.WhitespaceTokenizer()
  train = wp_titles_dataset(wptitles_train_path, tokenizer, grapheme_idx, MAX_STR_LEN, include_label=True)
  valid = wp_titles_dataset(wptitles_valid_path, tokenizer, grapheme_idx, MAX_STR_LEN, include_label=True)

  print(f"First train: {next(iter(train))}")
  print(f"First valid: {next(iter(valid))}")

  # Make the datum both input and output
  # train = train.map(lambda s: (s, s))
  # valid = valid.map(lambda s: (s, s))

  train = train.batch(BATCH, drop_remainder=True).map(prepare_batch, tf.data.AUTOTUNE)
  valid = valid.batch(BATCH, drop_remainder=True).map(prepare_batch, tf.data.AUTOTUNE)
  # test = test.batch(BATCH)


  # num_layers = 4
  d_model = 128
  # dff = 512
  num_heads = 8
  # dropout_rate = 0.1

  grapheme_count = len(grapheme_idx)
  print(f"grapheme_count: {grapheme_count}")

  train_in = train.map(lambda x,y: x)
  train_out = train.map(lambda x,y: y)

  # train_batch = next(iter(train))
  # valid_batch = next(iter(valid))

  # print(f"First train batch: {train_batch}")
  # print(f"First valid batch: {valid_batch}")

  # pos = PositionalEmbedding(
  #  vocab_size=grapheme_count,
  #  emb=d_model
  # )
  enc = Encoder(
   num_heads=num_heads,
   input_vocab_size=grapheme_count,
   emb=d_model,
  )
 # def __init__(self, *, num_heads: int, emb: int, output_vocab_size: int):
  dec = Decoder(
   num_heads=num_heads,
   emb=d_model,
   output_vocab_size=grapheme_count,
  )

  x = next(iter(train_in))

  # x_emb = pos(x)

  # print(x_emb)

  x_enc = enc(x)

  print(x_enc)

  x_dec = dec(x_enc)

  print(x_dec)

  # y = next(iter(train_out))

  # y_emb = pos(y)

  # print(y_emb)
