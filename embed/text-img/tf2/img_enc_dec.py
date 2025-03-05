import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, ReLU

from images import images_dataset

BATCH=50
W=128
H=128

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


class ConvNorm(tf.keras.layers.Layer):
 def __init__(self, filters: int,  shape: tuple, **kwargs):
  super().__init__()
  self.conv = Conv2D(filters, shape, **kwargs)
  # self.norm = tf.keras.layers.BatchNormalization()
  self.norm = tf.keras.layers.LayerNormalization()

 def call(self, x: tf.Tensor):
  x = self.conv(x)

  x = self.norm(x)
  return x

class ConvTransposeNorm(tf.keras.layers.Layer):
 def __init__(self, filters: int,  shape: tuple, **kwargs):
  super().__init__()
  self.conv = Conv2DTranspose(filters, shape, **kwargs)
  self.norm = tf.keras.layers.LayerNormalization()
  # self.norm = tf.keras.layers.BatchNormalization()

 def call(self, x: tf.Tensor):
  x = self.conv(x)

  x = self.norm(x)
  return x


class Encoder(tf.keras.layers.Layer):
 def __init__(self, *, emb: int):
  super().__init__()
  # self.dim_encoding = positional_encoding(length=max(W,H), depth=emb)

  self.convs = [
   ConvNorm(3, (65, 65)),
   ConvNorm(4, (33, 33)),
   ConvNorm(8, (17, 17)),
   ConvNorm(16, (9, 9)),
   ConvNorm(32, (5, 5)),
   ConvNorm(64, (3, 3)),
   ConvNorm(128, (2, 2)),
  ]

  self.emb_size = emb

 # x: an image
 # (batch, w, h, c)
 def call(self, x: tf.Tensor):
  # (batch, seq)
  # print(inputs)

  if type(x) == tuple:
   x,y = x

  # print(x.get_shape())
  batch, w, h, c = x.get_shape()

  # print(f"batch, w, h, c: {batch} {w} {h} {c}")

  # w_emb = self.dim_encoding[:w, :]
  # h_emb = self.dim_encoding[:h, :]

  # x = tf.image.resize(x, (W, H))

  for conv in self.convs:
   x = conv(x)

  # print(x.get_shape())
  # batch = x.get_shape()[0]

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
   # ConvTransposeNorm(64, 1, strides=8),# (batch, 16, 16, 64)
   # ConvTransposeNorm(32, 1, strides=2),# (batch, 32, 32, 32)
   # ConvTransposeNorm(16, 1, strides=2),# (batch, 64, 64, 16)
   # ConvNorm(16, 7, padding='same'),# (batch, 64, 64, 16)
   # ConvNorm(16, 7, padding='same'),# (batch, 64, 64, 16)
   # ConvTransposeNorm(8, 1, strides=2),# (batch, 128, 128, 8)
   # ConvNorm(3, 1),# (batch, 128, 128, 3)

   ConvTransposeNorm(64, 1, strides=8),# (batch, 16, 16, 64)
   ReLU(),
   ConvNorm(64, 5, padding='same'),# (batch, 16, 16, 64)
   ReLU(),

   ConvTransposeNorm(32, 1, strides=4),# (batch, 64, 64, 32)
   ReLU(),
   ConvNorm(32, 5, padding='same'),# (batch, 64, 64, 32)
   ReLU(),

   ConvTransposeNorm(16, 1, strides=2),# (batch, 128, 128, 16)
   ReLU(),
   ConvNorm(16, 7, padding='same'),# (batch, 128, 128, 16)
   ReLU(),
   ConvNorm(8, 7, padding='same'),# (batch, 128, 128, 8)
   ReLU(),
   ConvNorm(3, 7, padding='same'),# (batch, 128, 128, 3)

  ]

 # x: an embedding vector
 # (batch, emb)
 def call(self, x: tf.Tensor):
  print(f"img_dec x shape: {x.shape}")
  batch = x.get_shape()[0]
  
  # Use the emb vector as the channels of a 2 x 2 image nucleus
  x = tf.tile(x, (1, 4,))
  # (batch, 4*emb)

  x = tf.reshape(x, (batch, 2, 2, self.emb_size))
  # (batch, 2, 2, emb)

  #NOTE Do we need to add a 2d position embedding here to give it its bearings on the image?

  for conv in self.convs:
   x = conv(x)

  # x = tf.cast(tf.math.round(x), tf.int32)

  x = tf.keras.activations.sigmoid(x)

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
  
def load_datasets(train_path: str, valid_path: str) -> tuple[tf.data.Dataset, tf.data.Dataset]:
  # train = tf.keras.preprocessing.image_dataset_from_directory(
  #  train_path,
  #  labels=None,
  #  batch_size=BATCH,
  #  image_size=(W,H),
  #  follow_links=True,
  # )
  # valid = tf.keras.preprocessing.image_dataset_from_directory(
  #  valid_path,
  #  labels=None,
  #  batch_size=BATCH,
  #  image_size=(W,H),
  #  follow_links=True,
  # )

  train = images_dataset(train_path, 'train')
  valid = images_dataset(valid_path, 'valid')

  # Make the datum both input and output
  train = train.map(lambda s: (s, s))
  valid = valid.map(lambda s: (s, s))

  train = train.batch(BATCH, drop_remainder=True)
  valid = valid.batch(BATCH, drop_remainder=True)

  return train, valid

if __name__=="__main__":
  import argparse
  parser = argparse.ArgumentParser(
   description="Image encoder/decoder",
  )
  parser.add_argument('train_path')
  parser.add_argument('valid_path')
  parser.add_argument('output_path', help='Path of output Keras model file')
  args = parser.parse_args()

  
  train, valid = load_datasets(args.train_path, args.valid_path)

  print(f"First train: {next(iter(train))}")

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
  print(x[0].shape)
  print(x[1].shape)


  x_dec = m(x)

  print(x_dec[0].shape)
  print(x_dec[1].shape)


  # y = next(iter(train_out))

  # y_emb = pos(y)

  # print(y_emb)
  # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.98,
  #                                    epsilon=1e-9)
  # loss = tf.keras.losses.SparseCategoricalCrossentropy(
    # from_logits=True,
    # reduction='none'
  # )
  checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=args.output_path,
      monitor='val_loss',
      mode='min',
      save_best_only=True)

  m.compile(
   loss='mse',
   optimizer='adam',
   # run_eagerly=True,
  )

  m.fit(
   train,
   epochs=500,
   validation_data=valid,
   callbacks=[checkpoint_callback],
  )
