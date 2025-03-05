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
 def __init__(self, *, vocab_size: int, emb: int, length: int):
  super().__init__()
  self.emb_size = emb
  self.embedding = tf.keras.layers.Embedding(vocab_size, emb, mask_zero=True) 
  self.pos_encoding = positional_encoding(length=length, depth=emb)

 # def compute_mask(self, *args, **kwargs):
 #  return self.embedding.compute_mask(*args, **kwargs)

 def call(self, x: tf.Tensor):
  print(f"pos emb x shape: {x.shape}")
  # print(f"x.get_shape: {x.get_shape()}")
  length = x.get_shape()[1]
  # print(f"x.get_shape()[1]: {x.get_shape()[1]}")
  x = self.embedding(x)
  print(f"x shape 2: {x.shape}")
  # This factor sets the relative scale of the embedding and positonal_encoding.
  x *= tf.math.sqrt(tf.cast(self.emb_size, tf.float32))
  print(f"x shape 3: {x.shape}")
  emb_ = self.pos_encoding[tf.newaxis, :length, :]
  print(f"emb_ shape: {emb_.shape}")
  x = x + self.pos_encoding[tf.newaxis, :length, :]
  return x

def prepare_batch(x: tf.Tensor, y: tf.Tensor):
    x = x[:, :-1]# Drop end tokens

    y = y[:, 1:] # Drop start tokens

    return x, y

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

 def print(self, depth=0):
  tab = "\t" * depth
  print(f"{tab}MhaResLayerNorm")
  for w in self.mha.weights:
   print(f"{tab}\t{w.name}: {w.shape}")

 def size(self):
  size = 0
  for w in self.mha.weights:
   size += tf.size(w)

  return size

class FfResLayerNorm(tf.keras.layers.Layer):
 def __init__(self, *, emb: int, dff: int):
  super().__init__()
  #FIXME check why these consume so much memory
  self.ff1 = tf.keras.layers.Dense(dff, activation='relu')
  self.ff2 = tf.keras.layers.Dense(emb)
  self.layernorm = tf.keras.layers.LayerNormalization()
  self.add = tf.keras.layers.Add()

 def call(self, x: tf.Tensor):
  # (batch, seq, emb)

  ff = self.ff1(x)
  # (batch, seq, dff)

  ff = self.ff2(ff)
  # (batch, seq, emb)  

  x = self.add([x, ff])
  # (batch, seq, emb)

  x = self.layernorm(x)
  # (batch, seq, emb)

  return x

 def print(self, depth=0):
  tab = "\t" * depth
  print(f"{tab}FfResLayerNorm")
  print(f"{tab}\tff1: {self.ff1.kernel.shape}")
  print(f"{tab}\tff2: {self.ff2.kernel.shape}")
 
class TransformerLayer(tf.keras.layers.Layer):
 def __init__(self, *, num_heads: int, emb: int, dff: int):
  super().__init__()
  self.mha_l = MhaResLayerNorm(num_heads=num_heads, emb=emb)
  self.ff_l = FfResLayerNorm(emb=emb, dff=dff)

 def call(self, x: tf.Tensor):
  # (batch, seq, emb)
  return self.ff_l(self.mha_l(x))

 def print(self, depth=0):
  tab = "\t" * depth
  print(f"{tab}TransformerLayer")
  self.mha_l.print(depth=depth+1)
  self.ff_l.print(depth=depth+1)
 

class EncodingSummer(tf.keras.layers.Layer):
 def call(self, x: tf.Tensor):
  # (batch, seq, emb)
  x = tf.reduce_sum(x, 1)
  # (batch, emb)
  return x

 def print(self, depth=0):
  tab = "\t" * depth
  print(f"{tab}EncodingSummer")

# Encodes a sequence as an emb-length vector with no other context
class Encoder(tf.keras.layers.Layer):
 def __init__(self, *, layers: int, num_heads: int, input_vocab_size: int, emb: int, dff: int):
  super().__init__()
  self.pos = PositionalEmbedding(
   vocab_size=input_vocab_size,
   emb=emb,
   length=MAX_STR_LEN-1
  )
  self.enc = tf.keras.Sequential(
   [TransformerLayer(num_heads=num_heads, emb=emb, dff=dff)] * layers
   + [EncodingSummer()]
  )

 def call(self, x: tf.Tensor):
  print(f"txt enc x shape: {x.shape}")
  # (batch, seq)

  x = self.pos(x)

  # (batch, seq, emb)

  x = self.enc(x)

  # (batch, emb)

  return x

 def print(self, depth=0):
  tab = "\t" * depth
  print(f"{tab}Encoder")
  for l in self.enc.layers:
   l.print(depth=depth+1)
   

class Decoder(tf.keras.layers.Layer):
 def __init__(self, *, layers: int, num_heads: int, emb: int, dff: int, output_vocab_size: int):
  super().__init__()
  self.pos_encoding = positional_encoding(length=MAX_STR_LEN-1, depth=emb)
  self.dec = tf.keras.Sequential([TransformerLayer(num_heads=num_heads, emb=emb, dff=dff)] * layers)
  self.emb_size = emb
  self.emb_to_vocab = tf.keras.layers.Dense(output_vocab_size)
  self.softmax = tf.keras.layers.Softmax()

 def call(self, x: tf.Tensor):
  print(f"txt dec x shape: {x.shape}")
  # (batch, emb)

  # Add a cardinality 1 dimension at index 1
  s = list(x.get_shape())
  s.insert(1, 1)
  x = tf.reshape(x, s)
  # (batch, 1, emb)

  # Now repeat along the newly added dimension, one repetition per output seq location
  # MAX_STR_LEN-1 because we drop start/end tokens
  x = tf.repeat(x, MAX_STR_LEN-1, axis=1)
  # (batch, seq, emb)

  # Augment with a position encoding
  x = x + self.pos_encoding[tf.newaxis, :(MAX_STR_LEN-1), :]
  # (batch, seq, emb)

  x = self.dec(x)
  # (batch, seq, emb)

  x = self.emb_to_vocab(x)
  # (batch, seq, grapheme)

  x = self.softmax(x)
  # (batch, seq, grapheme)

  return x

 def print(self, depth=0):
  tab = "\t" * depth

  print(f"{tab}Decoder")
  for l in self.dec.layers:
   l.print(depth=depth+1)
  print(f"{tab}\temb_to_vocab: {self.emb_to_vocab.kernel.shape}")
   

class EncDec(tf.keras.Model):
 def __init__(self, *, layers: int, num_heads: int, emb:int, dff: int, vocab_size: int):
  super().__init__()
  self.enc = Encoder(
   layers=layers,
   num_heads=num_heads,
   input_vocab_size=vocab_size,
   emb=emb,
   dff=dff,
  )
  self.dec = Decoder(
   layers=layers,
   num_heads=num_heads,
   emb=emb,
   dff=dff,
   output_vocab_size=vocab_size,
  )

 # def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
  # x, y = inputs
 def call(self, x: tf.Tensor):
  x_enc = self.enc(x)
  return self.dec(x_enc)

 def print(self, depth=0):
  tab = "\t" * depth
  print("f{tab}EncDec")
  self.enc.print(depth=depth+1)
  self.dec.print(depth=depth+1)

 def size(self):
  return self.enc.size() + self.dec.size()
  
# Returns train, valid, grapheme_count
def load_datasets() -> tuple[tf.data.Dataset, tf.data.Dataset, GraphemeIdx]:
  # data_path = os.path.join(os.path.expanduser('~'), 'Data')
  data_path = '/blok/@data'

  train_path = os.path.join(data_path, 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_train.rand.gz')
  valid_path = os.path.join(data_path, 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_valid.rand.gz')
  # wptitles_test_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_test.rand.gz')

  grapheme_idx = load_grapheme_idx()
  # print(grapheme_idx)

  tokenizer = tft.WhitespaceTokenizer()
  train = wp_titles_dataset(train_path, tokenizer, grapheme_idx, MAX_STR_LEN, include_label=True)
  valid = wp_titles_dataset(valid_path, tokenizer, grapheme_idx, MAX_STR_LEN, include_label=True)

  # print(f"First train: {next(iter(train))}")
  # print(f"First valid: {next(iter(valid))}")

  # Make the datum both input and output
  # train = train.map(lambda s: (s, s))
  # valid = valid.map(lambda s: (s, s))

  train = train.batch(BATCH, drop_remainder=True).map(prepare_batch, tf.data.AUTOTUNE)
  valid = valid.batch(BATCH, drop_remainder=True).map(prepare_batch, tf.data.AUTOTUNE)
  # test = test.batch(BATCH)

  return train, valid, grapheme_idx

class TextCallback(tf.keras.callbacks.Callback):
 def __init__(self, output_dir: str, valid: tf.data.Dataset, grapheme_idx: GraphemeIdx):
  self.output_dir = output_dir
  self.valid = valid
  self.grapheme_idx = grapheme_idx

 def on_epoch_end(self, epoch, logs, *args, **kwargs):
  with open(f"{self.output_dir}/texts-it-{epoch}", 'w') as w:
   s, _ = next(iter(self.valid))

   original_text = self.grapheme_idx.unindex_txts(s.numpy().tolist())

   result_probs = self.model(s)

   # print(result_probs.get_shape())

   result = tf.argmax(result_probs, axis=2)

   # print(result.get_shape())

   l = result.numpy().tolist()

   # print(l)

   text = self.grapheme_idx.unindex_txts(l)

   for i, t in enumerate(text):
    w.write(original_text[i])
    w.write('\t')
    w.write(t)
    w.write('\n')

  

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(
   description="Text encoder/decoder",
  )
  # parser.add_argument('train_path')
  # parser.add_argument('valid_path')
  parser.add_argument('output_dir')
  args = parser.parse_args()

  os.makedirs(args.output_dir, exist_ok=True)


  train, valid, grapheme_idx = load_datasets()

  layers = 2
  d_model = 128
  dff = 4 * d_model
  num_heads = 8
  # dropout_rate = 0.1

  grapheme_count = len(grapheme_idx)
  # print(f"grapheme_count: {grapheme_count}")

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
   layers=layers,
   num_heads=num_heads,
   emb=d_model,
   dff=dff,
   vocab_size=grapheme_count,
  )

  x = next(iter(train_in))

  # x_emb = pos(x)

  # print(x_emb)

  # x_enc = enc(x)

  # print(x_enc)
  # print(f"x_enc shape: {x_enc.get_shape()}")

  # x_dec = dec(x_enc)

  x_dec = m(x)


  # print(x_dec)

  # y = next(iter(train_out))

  # y_emb = pos(y)

  # print(y_emb)
  # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98,
  #                                    epsilon=1e-9)

  callbacks = [
   tf.keras.callbacks.ModelCheckpoint(
      filepath=f"{args.output_dir}/best.keras",
      monitor='val_loss',
      mode='min',
      save_best_only=True
   ),
   TextCallback(args.output_dir, valid, grapheme_idx),
  ]

  m.compile(
   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
   optimizer='sgd',
   run_eagerly=True,
  )

  m.print()
  # print(m.size()) #FIXME

  m.fit(train, epochs=500, validation_data=valid, callbacks=callbacks)

  # with open('/tmp/texts', 'w') as w:
  #  s, _ = next(iter(valid))

  #  original_text = grapheme_idx.unindex_txts(s.numpy().tolist())

  #  result_probs = m(s)

  #  # print(result_probs.get_shape())

  #  result = tf.argmax(result_probs, axis=2)

  #  # print(result.get_shape())

  #  l = result.numpy().tolist()

  #  # print(l)

  #  text = grapheme_idx.unindex_txts(l)

  #  for i, t in enumerate(text):
  #   w.write(original_text[i])
  #   w.write('\t')
  #   w.write(t)
  #   w.write('\n')
