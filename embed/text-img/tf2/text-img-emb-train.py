from statistics import fmean

from img_enc_dec import Encoder as ImgEncoder, Decoder as ImgDecoder, EncDec as ImgEncDec, load_datasets as load_img_datasets, W, H
from text_enc_dec import Encoder as TextEncoder, Decoder as TextDecoder, EncDec as TextEncDec, MAX_STR_LEN, prepare_batch as prepare_text_batch, load_datasets as load_text_datasets

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

EMB=128

class Conv(tf.keras.layers.Layer):
 def __init__(self, *args, **kwargs):
  self.bn = tf.keras.layers.BatchNormalization()
  self.relu = tf.keras.layers.ReLU()
  self.conv = tf.keras.layers.Conv2d(
   *args,
   **kwargs,
  )

 def call(self, x: tf.Tensor):
  x = self.bn(x)
  x = self.conv(x)
  x = self.relu(x)
  return x

# A "binary operator" for text sequences that partially follows "Pervasive Attention":
# https://arxiv.org/pdf/1808.03867
#
# We don't use the DenseNet-style all-preceding-layer inputs
#
# We also don't mask "future" output tokens as we are not doing translation; this is meant as a binary operator by which to define a group of texts
class TextBinOp(tf.keras.layers.Layer):
 def __init__(self, *, emb: int):
  self.emb = emb

  # Use large convolutions to somewhat compensate for the lack of DenseNet-style connections
  # This way we get longer-distance connections with few layers, but higher computational requirements of course
  self.convs = [
   Conv(2 * emb, (9, 9), padding='same'),
   Conv(2 * emb, (9, 9), padding='same'),
   Conv(2 * emb, (9, 9), padding='same'),
   Conv(emb, (9, 9), padding='same'),
   Conv(emb, (9, 9), padding='same'),
   Conv(emb, (9, 9), padding='same'),
  ]

 # x: tokens
 #    (batch, seq, emb)
 # y: tokens
 #    (batch, seq, emb)
 #
 # Asserts x.shape == y.shape
 #
 # returns token indices, shape (batch, seq, emb) with the same shape as the inputs
 def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  assert(x.shape == y.shape)
  assert(x.shape[2] == self.emb)
  assert(y.shape[2] == self.emb)

  # Build the 2d translation matrix
  # The channels are a concatenation of the source and target embeddings
  # i.e. join x and y such that m[:, i, j] is a concatenation of the channels of x[:, i] and y[:, j]
  # (batch, seq, seq, 2*emb)
  xs = x.get_shape()
  s = (xs[0], xs[1], xs[1], 2*xs[2])

  m = tf.zeros(s)
  for i in range(xs[1]):
   x_c = x[:, i]
   for j in range(xs[1]):
    y_c = y[:, j]

    # Concatenate the channels of x and y
    m[:, i, j] = tf.concat([x_c, y_c], 1)

  # Ignore the construction of the matrix when we backpropagate
  m = tf.stop_gradient(m)

  # (batch, seq, seq, 2*emb)

  # Apply convolutions
  # These are naive - no DenseNet
  for c in self.convs:
   m = c(m)

  # (batch, seq, seq, emb)

  # Max pool the source seq away
  m = tf.reduce_max(m, 1)
  # (batch, seq, emb)

  return m


# A "binary operator" for images
#
# This is meant to define a "group" of images
class ImgBinOp(tf.keras.layers.Layer):
 def __init__(self, *, c: int):
  self.c = c

  # Use large convolutions to somewhat compensate for the lack of DenseNet-style connections
  # This way we get longer-distance connections with few layers, but higher computational requirements of course

  # Initial high-dimensionality large filters
  self.convs = 2 * [
    Conv(2 * c, (9, 9), padding='same')
  ]

  # Drop one channel at a time
  for i in range(c):
   self.convs.append(
    Conv(2 * c - i, (9, 9), padding='same')
   )

  # Trailing still-large low-dimensionality filters
  self.convs.extend(2 * [
   Conv(c, (9, 9), padding='same'),
  ])

 # x: images
 #    (batch, w, h, c)
 # y: images
 #    (batch, w, h, c)
 #
 # Asserts x.shape == y.shape
 #
 # returns images, shape (batch, w, h, c) with the same shape as the inputs
 def call(self, x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
  assert(x.get_shape() == y.get_shape())
  assert(x.get_shape()[-1] == self.c)
  assert(y.get_shape()[-1] == self.c)

  # Concatenate the inputs across channels
  z = tf.concat([x, y], 3)
  # (batch, w, h, 2*c)

  for c in self.convs:
   z = c(z)

  return z

# Image autoencoder training step
#
# x shape: (batch, w, h, c)
@tf.function
def img_enc_dec_train_step(*,
 opt: tf.keras.optimizers.Optimizer,
 loss_fn: tf.keras.Loss,
 img_enc: ImgEncoder,
 img_dec: ImgDecoder,
 x: tf.Tensor,
 loss_fn = tf.keras.losses.MeanSquaredError(),
 train = True,
):
 with tf.GradientTape as tape:
  y = img_dec(img_enc(x))
  loss = loss_fn(x, y)

 if train:
  enc_grads = tape.gradient(loss, img_enc.trainable_weights)
  dec_grads = tape.gradient(loss, img_dec.trainable_weights)
  opt.apply_gradients(zip(enc_grads, img_enc.trainable_weights))
  opt.apply_gradients(zip(dec_grads, img_dec.trainable_weights))

 return loss.numpy()
 
# Text autoencoder training step
#
# x shape: (batch, seq)
@tf.function
def txt_enc_dec_train_step(*,
 opt: tf.keras.optimizers.Optimizer,
 txt_enc: TextEncoder,
 txt_dec: TextDecoder,
 x: tf.Tensor,
 loss_fn = tf.keras.losses.SparseCategoricalCrossEntropy(),
 train = True,
):
 with tf.GradientTape as tape:
  y = txt_dec(txt_enc(x))
  loss = loss_fn(x, y)

 if train:
  enc_grads = tape.gradient(loss, txt_enc.trainable_weights)
  dec_grads = tape.gradient(loss, txt_dec.trainable_weights)
  opt.apply_gradients(zip(enc_grads, txt_enc.trainable_weights))
  opt.apply_gradients(zip(dec_grads, txt_dec.trainable_weights))

 return loss.numpy()

# Image autoencoder training step, starting from embedding
#
# x shape: (batch, emb)
@tf.function
def img_dec_enc_train_step(*,
 opt: tf.keras.optimizers.Optimizer,
 img_enc: ImgEncoder,
 img_dec: ImgDecoder,
 x: tf.Tensor,
 loss_fn = tf.keras.losses.MeanSquaredError(),
 train = True,
):
 with tf.GradientTape as tape:
  y = img_enc(img_dec(x))
  loss = loss_fn(x, y)

 if train:
  enc_grads = tape.gradient(loss, img_enc.trainable_weights)
  dec_grads = tape.gradient(loss, img_dec.trainable_weights)
  opt.apply_gradients(zip(enc_grads, img_enc.trainable_weights))
  opt.apply_gradients(zip(dec_grads, img_dec.trainable_weights))

 return loss.numpy() 

# Text autoencoder training step, starting from embedding
#
# x shape: (batch, emb)
@tf.function
def txt_dec_enc_train_step(*,
 opt: tf.keras.optimizers.Optimizer,
 txt_enc: TextEncoder,
 txt_dec: TextDecoder,
 x: tf.Tensor,
 loss_fn = tf.keras.losses.MeanSquaredError(),
 train = True,
):
 with tf.GradientTape as tape:
  y = txt_enc(txt_dec(x))
  loss = loss_fn(x, y)

 if train:
  enc_grads = tape.gradient(loss, txt_enc.trainable_weights)
  dec_grads = tape.gradient(loss, txt_dec.trainable_weights)
  opt.apply_gradients(zip(enc_grads, txt_enc.trainable_weights))
  opt.apply_gradients(zip(dec_grads, txt_dec.trainable_weights))

 return loss.numpy() 


# Image binary operation training step
#
# Starts in embedding space, translates to image space, applies binary op, then moves back to embedding space
#
# x shape: ((batch, emb), (batch, emb)), (batch, emb)
@tf.function
def img_bin_op_train_step(*,
 opt: tf.keras.optimizers.Optimizer,
 img_enc: ImgEncoder,
 img_dec: ImgDecoder,
 img_bin_op: ImgBinOp,
 inputs: tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
 loss_fn = tf.keras.losses.MeanSquaredError(),
 train = True,
):
 (x,y),z = inputs
 with tf.GradientTape as tape:
  x_img = img_dec(x)
  y_img = img_dec(y)
  z_img_pred = img_bin_op(x_img, y_img)
  z_pred = img_enc(z_img_pred)

  loss = loss_fn(z, z_pred)

 if train:
  enc_grads = tape.gradient(loss, img_enc.trainable_weights)
  dec_grads = tape.gradient(loss, img_dec.trainable_weights)
  bin_op_grads = tape.gradient(loss, img_bin_op.trainable_weights)
  opt.apply_gradients(zip(enc_grads, img_enc.trainable_weights))
  opt.apply_gradients(zip(dec_grads, img_dec.trainable_weights))
  opt.apply_gradients(zip(bin_op_grads, img_bin_op.trainable_weights))

 return loss.numpy()

# Image binary operation training step
#
# Starts in embedding space, translates to text space, applies binary op, then moves back to embedding space
#
# x shape: ((batch, emb), (batch, emb)), (batch, emb)
@tf.function
def txt_bin_op_train_step(*,
 opt: tf.keras.optimizers.Optimizer,
 txt_enc: TxtEncoder,
 txt_dec: TxtDecoder,
 txt_bin_op: TxtBinOp,
 inputs: tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor],
 loss_fn = tf.keras.losses.MeanSquaredError(),
 train = True,
):
 (x,y),z = inputs
 with tf.GradientTape as tape:
  x_txt = txt_dec(x)
  y_txt = txt_dec(y)
  z_txt_pred = txt_bin_op(x_txt, y_txt)
  z_pred = txt_enc(z_txt_pred)

  loss = loss_fn(z, z_pred)

 if train:
  enc_grads = tape.gradient(loss, txt_enc.trainable_weights)
  dec_grads = tape.gradient(loss, txt_dec.trainable_weights)
  bin_op_grads = tape.gradient(loss, txt_bin_op.trainable_weights)
  opt.apply_gradients(zip(enc_grads, txt_enc.trainable_weights))
  opt.apply_gradients(zip(dec_grads, txt_dec.trainable_weights))
  opt.apply_gradients(zip(bin_op_grads, txt_bin_op.trainable_weights))

 return loss.numpy()

def train(*,
 opt: tf.keras.optimizers.Optimizer,
 epochs: int,
 steps_per_epoch: int,
 valid_steps_per_epoch: int,
 img_enc: ImgEncoder,# imb -> emb
 img_dec: ImgDecoder,# emb -> img
 txt_enc: TextEncoder,# txt -> emb
 txt_dec: TextDecoder,# emb -> txt
 img_bin_op: ImgBinOp,# img ⊕  img -> img 
 txt_bin_op: TextBinOp,# txt ⊞ txt -> txt
 emb_dataset: tf.data.Dataset,# infinitely looped, unabeled
 img_train: tf.data.Dataset,# infinitely looped, unlabeled
 txt_train: tf.data.Dataset,# infinitely looped, unlabeled
 img_valid: tf.data.Dataset,# infinitely looped, unlabeled
 txt_valid: tf.data.Dataset,# infinitely looped, unlabeled
):
#FIXME randomize image order

 emb_sum = emb_dataset.zip(emb_dataset).map(lambda a,b: a+b)
 emb_sum_iter = iter(emb_sum)
 
 emb_iter = iter(emb_dataset)
 img_train_iter = iter(img_train)
 txt_train_iter = iter(txt_train)
 img_valid_iter = iter(img_valid)
 txt_valid_iter = iter(txt_valid)

 for e in range(epochs):
  print(f"Epoch {e}/{epochs}")

  losses = defaultdict(list)
  
  for s in range(steps_per_epoch):
   # img enc/dec roundtrip
   losses['img_enc_dec'].append(img_enc_dec_train_step(
    opt=opt,
    img_enc=img_enc,
    img_dec=img_dec,
    x=next(img_train_iter),
   ))

   # text enc/dec roundtrip
   losses['txt_enc_dec'].append(txt_enc_dec_train_step(
    opt=opt,
    txt_enc=txt_enc,
    txt_dec=txt_dec,
    x=next(txt_train_iter),
   ))

   # img dec/enc roundrip (starting from emb)
   losses['img_dec_enc'].append(img_dec_enc_train_step(
    opt=opt,
    img_enc=img_enc,
    img_dec=img_dec,
    x=next(img_train_iter),
   ))

   # text dec/enc roundtrip (starting from emb)
   losses['txt_dec_enc'].append(txt_dec_enc_train_step(
    opt=opt,
    txt_enc=txt_enc,
    txt_dec=txt_dec,
    x=next(txt_train_iter),
   ))

   # img binop roundtrip
   losses['img_bin_op'].append(img_bin_op_train_step(
    opt=opt,
    img_enc=img_enc,
    img_dec=img_dec,
    img_bin_op=img_bin_op,
    x=next(emb_sum_iter),
   ))

   # text binop roundtrip
   losses['txt_bin_op'].append(txt_bin_op_train_step(
    opt=opt,
    txt_enc=txt_enc,
    txt_dec=txt_dec,
    txt_bin_op=txt_bin_op,
    x=next(emb_sum_iter),
   ))

   loss_keys = list(sorted(losses.keys()))
   for k in loss_keys:
    l = fmean(losses[k])
    
    print(f"{k}: {l}")

  valid_losses = defaultdict(list)

  for s in range(valid_steps_per_epoch):
   # img enc/dec roundtrip
   valid_losses['img_enc_dec'].append(img_enc_dec_train_step(
    opt=opt,
    img_enc=img_enc,
    img_dec=img_dec,
    x=next(img_valid_iter),
    train=False,
   ))

   # text enc/dec roundtrip
   valid_losses['txt_enc_dec'].append(txt_enc_dec_train_step(
    opt=opt,
    txt_enc=txt_enc,
    txt_dec=txt_dec,
    x=next(txt_valid_iter),
    train=False,
   ))

   # img dec/enc roundrip (starting from emb)
   valid_losses['img_dec_enc'].append(img_dec_enc_train_step(
    opt=opt,
    img_enc=img_enc,
    img_dec=img_dec,
    x=next(img_valid_iter),
    train=False,
   ))

   # text dec/enc roundtrip (starting from emb)
   valid_losses['txt_dec_enc'].append(txt_dec_enc_train_step(
    opt=opt,
    txt_enc=txt_enc,
    txt_dec=txt_dec,
    x=next(txt_valid_iter),
    train=False,
   ))

   # img binop roundtrip
   valid_losses['img_bin_op'].append(img_bin_op_train_step(
    opt=opt,
    img_enc=img_enc,
    img_dec=img_dec,
    img_bin_op=img_bin_op,
    x=next(emb_sum_iter),
    train=False,
   ))

   # text binop roundtrip
   valid_losses['txt_bin_op'].append(txt_bin_op_train_step(
    opt=opt,
    txt_enc=txt_enc,
    txt_dec=txt_dec,
    txt_bin_op=txt_bin_op,
    x=next(emb_sum_iter),
    train=False,
   ))

   valid_loss_keys = list(sorted(valid_losses.keys()))
   for k in valid_loss_keys:
    l = fmean(valid_losses[k])
    
    print(f"valid {k}: {l}")


def _gen_emb_dataset(*, batch: int, emb: int, dtype=tf.float32):
 while True:
  yield tf.random.uniform((batch, emb), dtype=dtype)

def emb_dataset(*, batch: int, emb: int, dtype=tf.float32):
 gen = lambda: _gen_emb_dataset(batch=batch, emb=emb, dtype=dtype)

 return tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(batch,emb,), dtype=type))


if __name__=="__main__":
  img_train, img_valid = load_img_datasets()
  txt_train, txt_valid, grapheme_count = load_text_datasets()

  img_train = img_train.repeat()
  img_valid = img_train.repeat()
  txt_train = txt_train.repeat()
  txt_valid = txt_valid.repeat()

  num_heads = 8

  opt = Adam()
  img_enc = ImgEncoder(emb=EMB)
  img_dec = ImgDecoder(emb=EMB, w=W, h=H)
  txt_enc = TextEncoder(num_heads=num_heads, input_vocab_size=grapheme_count, emb=EMB)
  txt_dec = TextDecoder(num_heads=num_heads, emb=EMB, output_vocab_size=grapheme_count)

  emb_dataset_ = emb_dataset(batch=BATCH, emb=EMB)

  train(
   opt=opt,
   epochs=20,
   steps_per_epoch=1000,
   valid_steps_per_epoch=100,
   img_enc=img_enc,
   img_dec=img_dec,
   txt_enc=txt_enc,
   txt_dec=txt_dec,
   emb_dataset=emb_dataset_,
   img_train=img_train,
   txt_train=txt_train,
   img_valid=img_valid,
   txt_valid=txt_valid,
  )
