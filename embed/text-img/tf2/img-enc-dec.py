import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D

from images import images_dataset

BATCH=10
W=256
H=256

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

# class PositionalEmbedding(tf.keras.layers.Layer):
#  def __init__(self, *, vocab_size: int, emb: int):
#   super().__init__()
#   self.emb_size = emb
#   self.embedding = tf.keras.layers.Embedding(vocab_size, emb, mask_zero=True) 
#   self.pos_encoding = positional_encoding(length=2048, depth=emb)

#  def compute_mask(self, *args, **kwargs):
#   return self.embedding.compute_mask(*args, **kwargs)

#  def call(self, x: tf.Tensor):
#   # print(f"x.get_shape: {x.get_shape()}")
#   length = x.get_shape()[1]
#   # print(f"x.get_shape()[1]: {x.get_shape()[1]}")
#   x = self.embedding(x)
#   # This factor sets the relative scale of the embedding and positonal_encoding.
#   x *= tf.math.sqrt(tf.cast(self.emb_size, tf.float32))
#   x = x + self.pos_encoding[tf.newaxis, :length, :]
#   return x


class ResConvNorm(tf.keras.layers.Layer):
 def __init__(self, filters: int,  shape: tuple, **kwargs):
  super().__init__()
  self.conv = Conv2D(filters, shape, **kwargs)
  self.layernorm = tf.keras.layers.LayerNormalization()

 def call(self, x: tf.Tensor):
  original = x
  x = self.conv(x)

  x = self.layernorm(x)
  return x


class Encoder(tf.keras.layers.Layer):
 def __init__(self, *, emb: int):
  super().__init__()
  self.dim_encoding = positional_encoding(length=2048, depth=emb)

  self.convs = [
   ResConvNorm(3, (65, 65)),
   ResConvNorm(4, (33, 33)),
   ResConvNorm(8, (17, 17)),
   ResConvNorm(16, (9, 9)),
   ResConvNorm(32, (5, 5)),
   ResConvNorm(64, (3, 3)),
   ResConvNorm(128, (2, 2)),
  ]

  self.emb_size = emb

 # x: an image
 # (batch, w, h, c)
 def call(self, x: tf.Tensor):
  # (batch, seq)

  # print(x.get_shape())
  batch, w, h, c = x.get_shape()

  # print(f"batch, w, h, c: {batch} {w} {h} {c}")

  # w_emb = self.dim_encoding[:w, :]
  # h_emb = self.dim_encoding[:h, :]

  x = tf.image.resize(x, (128, 128))

  for conv in self.convs:
   x = conv(x)

  # print(x.get_shape())
  batch = x.get_shape()[0]

  x = tf.reshape(x, (batch, self.emb_size,))

  # return w_emb + x#  h_emb + 
  # TODO Maybe add width and depth positional encoding
  return x

class Decoder(tf.keras.layers.Layer):
 def __init__(self, *, emb: int, w: int, h: int):
  super().__init__()
  self.emb_size = emb
  self.w = w
  self.h = h
  self.convs = [
   ResConvNorm(64, (64, 64), padding='same'),
   ResConvNorm(32, (32, 32), padding='same'),
   ResConvNorm(16, (16, 16), padding='same'),
   ResConvNorm(16, (8, 8), padding='same'),
   ResConvNorm(16, (4, 4), padding='same'),
   ResConvNorm(8, (2, 2), padding='same'),
   ResConvNorm(3, (1, 1), padding='same'),
  ]

 # x: an embedding vector
 # (batch, emb)
 def call(self, x: tf.Tensor):
  batch = x.get_shape()[0]
  
  # Use the emb vector as the channels of a w x h sized image
  x = tf.tile(x, (1, self.w * self.h,))
  # (batch, w*h*emb)

  x = tf.reshape(x, (batch, self.w, self.h, self.emb_size))
  # (batch, w, h, emb)

  #NOTE Do we need to add a 2d position embedding here to give it its bearings on the image?

  for conv in self.convs:
   x = conv(x)

  return x

class EncDec(tf.keras.Model):
 def __init__(self, *, emb:int, w:int, h:int):
  super().__init__()

  self.enc = Encoder(
   emb=emb,
  )
  self.dec = Decoder(
   emb=emb,
   w=w,
   h=h
  )

 def call(self, x: tf.Tensor):
  x_enc = self.enc(x)
  return self.dec(x_enc)
  

if __name__=="__main__":
  home_dir = os.path.expanduser('~')

  train_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'wikimedia-commons-downscaled2_mini', 's100', 'q90')

  # train = images_dataset(path, 'train')
  # valid = images_dataset(path, 'valid')

  train = tf.keras.preprocessing.image_dataset_from_directory(
   train_path,
   labels=None,
   batch_size=BATCH,
   image_size=(W,H),
  )

  print(f"First train: {next(iter(train))}")
  # print(f"First valid: {next(iter(valid))}")

  # Make the datum both input and output
  # train = train.map(lambda s: (s, s))
  # valid = valid.map(lambda s: (s, s))

  # train = train.batch(BATCH, drop_remainder=True)
  # valid = valid.ragged_batch(BATCH, drop_remainder=True)
  # test = test.ragged_batch(BATCH, drop_remainder=True)

  # num_layers = 4
  d_model = 128
  # dff = 512
  # num_heads = 8
  # dropout_rate = 0.1


  m = EncDec(
   emb=d_model,
   w=W,
   h=H
  )

  x = next(iter(train))

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

  m.fit(
   train,
   epochs=20,
   # validation_data=valid
  )
