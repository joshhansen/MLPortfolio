import os

import numpy as np
import tensorflow as tf
import tensorflow_text as tft

from grapheme_idx import GraphemeIdx, load_grapheme_idx
from wptitles import wp_titles_dataset

BATCH=1250
MAX_STR_LEN = 128

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
  # print(f"Enc x shape: {x.get_shape()}")
  # (batch, seq, emb)
  attn = self.mha(
   query=x,
   value=x,
   key=x
  )
  # (batch, seq, emb)
  x = self.add([x, attn])
  # print(f"Enc x shape2: {x.get_shape()}")
  # (batch, seq, emb)
  x = tf.reduce_sum(x, 1)
  # (batch, emb)
  # print(f"Enc x shape3: {x.get_shape()}")
  x = self.layernorm(x)
  # (batch, emb)
  # print(f"Enc x shape4: {x.get_shape()}")
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

class Dec(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int):
  super().__init__()
  self.mha = tf.keras.layers.MultiHeadAttention(
   num_heads=num_heads,
   key_dim=emb,
  )
  self.layernorm = tf.keras.layers.LayerNormalization()
  self.add = tf.keras.layers.Add()


 def call(self, x: tf.Tensor):
  attn = self.mha(
   query=x,
   value=x,
   key=x,
  )
  x = self.add([x, attn])
  x = self.layernorm(x)

  return x
  
 

class Decoder(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int, output_vocab_size: int):
  super().__init__()
  self.pos_encoding = positional_encoding(length=2048, depth=emb)
  self.dec = Dec(num_heads=num_heads, emb=emb)
  self.emb_size = emb
  self.emb_to_vocab = tf.keras.layers.Dense(output_vocab_size)

 def call(self, x: tf.Tensor):
  # (batch, emb)
  # print(f"Dec x shape: {x.get_shape()}")

  #TODO Does manually broadcasting make sense? Is it needed?
  # x = tf.repeat(x, MAX_STR_LEN - 1, 1)
  x = tf.tile(x, [1, MAX_STR_LEN - 1])
  # (batch, seq*emb)
  # print(f"Dec x shape2: {x.get_shape()}")

  batch = x.get_shape()[0]

  #TODO Check if this makes sense!
  x = tf.reshape(x, (batch, MAX_STR_LEN - 1, self.emb_size))
  # (batch, seq, emb)
  # print(f"Dec x shape2': {x.get_shape()}")
  # print(x)

  # Augment with a position encoding
  length = x.get_shape()[1]
  x = x + self.pos_encoding[tf.newaxis, :length, :]

  x = self.dec(x)

  x = self.emb_to_vocab(x)

  # (batch, seq, vocab)
  # print(f"Dec x shape4: {x.get_shape()}")

  # x = tf.math.argmax(x, 2)

  # (batch, seq) ???
  # print(f"Dec x shape5: {x.get_shape()}")

  return x

class EncDec(tf.keras.Model):
 def __init__(self, *, num_heads: int, emb:int, vocab_size: int):
  super().__init__()
  self.enc = Encoder(
   num_heads=num_heads,
   input_vocab_size=vocab_size,
   emb=emb,
  )
  self.dec = Decoder(
   num_heads=num_heads,
   emb=emb,
   output_vocab_size=vocab_size,
  )

 # def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
  # x, y = inputs
 def call(self, x: tf.Tensor):

  x_enc = self.enc(x)
  return self.dec(x_enc)
  
# Returns train, valid, grapheme_count
def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, int]:
  # data_path = os.path.join(os.path.expanduser('~'), 'Data')
  data_path = '/blok/@data'

  train_path = os.path.join(data_path, 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_train.rand.gz')
  valid_path = os.path.join(data_path, 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_valid.rand.gz')
  # wptitles_test_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_test.rand.gz')

  grapheme_idx = load_grapheme_idx()
  print(grapheme_idx)

  tokenizer = tft.WhitespaceTokenizer()
  train = wp_titles_dataset(train_path, tokenizer, grapheme_idx, MAX_STR_LEN, include_label=True)
  valid = wp_titles_dataset(valid_path, tokenizer, grapheme_idx, MAX_STR_LEN, include_label=True)

  print(f"First train: {next(iter(train))}")
  print(f"First valid: {next(iter(valid))}")

  # Make the datum both input and output
  # train = train.map(lambda s: (s, s))
  # valid = valid.map(lambda s: (s, s))

  train = train.batch(BATCH, drop_remainder=True).map(prepare_batch, tf.data.AUTOTUNE)
  valid = valid.batch(BATCH, drop_remainder=True).map(prepare_batch, tf.data.AUTOTUNE)
  # test = test.batch(BATCH)

  return train, valid, len(grapheme_idx)

if __name__=="__main__":

  train, valid, grapheme_count = load_datasets()

  # num_layers = 4
  d_model = 128
  # dff = 512
  num_heads = 8
  # dropout_rate = 0.1

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
 #  enc = Encoder(
 #   num_heads=num_heads,
 #   input_vocab_size=grapheme_count,
 #   emb=d_model,
 #  )
 # # def __init__(self, *, num_heads: int, emb: int, output_vocab_size: int):
 #  dec = Decoder(
 #   num_heads=num_heads,
 #   emb=d_model,
 #   output_vocab_size=grapheme_count,
 #  )

  m = EncDec(
   num_heads=num_heads,
   emb=d_model,
   input_vocab_size=grapheme_count,
   output_vocab_size=grapheme_count,
  )

  x = next(iter(train_in))

  # x_emb = pos(x)

  # print(x_emb)

  # x_enc = enc(x)

  # print(x_enc)
  # print(f"x_enc shape: {x_enc.get_shape()}")

  # x_dec = dec(x_enc)

  x_dec = m(x)


  print(x_dec)

  # y = next(iter(train_out))

  # y_emb = pos(y)

  # print(y_emb)
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
    # from_logits=True,
    # reduction='none'
  )

  m.compile(
   loss=loss,
   optimizer=optimizer,
   run_eagerly=True,
  )

  m.fit(train, epochs=20, validation_data=valid)
