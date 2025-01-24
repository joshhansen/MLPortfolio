from statistics import fmean

from img_enc_dec import Encoder as ImgEncoder, Decoder as ImgDecoder, EncDec as ImgEncDec, load_datasets as load_img_datasets, W, H
from text_enc_dec import Encoder as TextEncoder, Decoder as TextDecoder, EncDec as TextEncDec, MAX_STR_LEN, prepare_batch as prepare_text_batch, load_datasets as load_text_datasets

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

EMB=128

# def _gen_random_dataset(width: int, len: int include_label=False):
#   for _ in range(len):
#     value = tf.random.uniform((width,))

#     if include_label:
#       yield value, value
#     else:
#       yield value


# def random_dataset(width: int, len: int, include_label=False):
#     gen = lambda: _gen_random_dataset(width, len, include_label)

#     if include_label:
#         return tf.data.Dataset.from_generator(gen, output_signature=(
#             tf.TensorSpec(shape=(width,), dtype=tf.float32),
#             tf.TensorSpec(shape=(width,), dtype=tf.float32)
#         ))

#     return tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(width,), dtype=tf.float32))

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

# Capture a relationship of isomorphism between two groups
#
# The dataset should provide (a,b), c with c = a*b
#
# So the binary operation on the source group is given; we are training the src->dest and dest->src mappings and the binary operation in the destination
class BinOpRoundTrip(tf.keras.layers.Layer):
 def __init__(self, *, src_to_dest: tf.keras.layers.Layer, dest_to_src: tf.keras.layers.Layer, dest_bin_op: tf.keras.layers.Layer):
  self.src_to_dest = src_to_test
  self.dest_to_src = dest_to_src
  self.dest_bin_op = dest_bin_op

 # The dataset should have (a,b) features and c labels, where a * b = c captures binary operation * on the source domain
 def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
  src_a, src_b = inputs

  dest_a = self.src_to_dest(src_a)
  dest_b = self.src_to_dest(src_b)
  dest_c = self.dest_bin_op(dest_a, dest_b)

  pred_src_c = self.dest_to_src(dest_c)

  return pred_src_c

class RoundTripBase(tf.keras.layers.Layer):
  def __init__(self, *, img_enc: ImgEncoder, img_dec: ImgDecoder, text_enc: TextEncoder, text_dec: TextDecoder):
    super().__init__()

    self.img_enc = img_enc
    self.img_dec = img_dec
    self.text_enc = text_enc
    self.text_dec = text_dec

class Text2Emb2Img2Emb2Text(RoundTripBase):
  def call(self, text: tf.Tensor):
    return self.text_dec(self.img_enc(self.img_dec(self.text_enc(text))))

class Img2Emb2Text2Emb2Img(RoundTripBase):
  def call(self, img: tf.Tensor):
    return self.img_dec(self.text_enc(self.text_dec(self.img_enc(img))))
    
class EmbeddingRoundTripper(tf.keras.Model):
  def __init__(self, *, num_heads: int, vocab_size: int):
    super().__init__()

    self.img_enc_dec = ImgEncDec(EMB, W, H)
    self.text_enc_dec = TextEncDec(num_heads, EMB, vocab_size)

    self.img_roundtrip = Img2Emb2Text2Emb2Img(
      img_enc = self.img_enc_dec.enc,
      img_dec = self.img_enc_dec.dec,
      text_enc = self.text_enc_dec.enc,
      text_dec = self.text_enc_dec.dec,
    )
    self.text_roundtrip = Text2Emb2Img2Emb2Text(
      img_enc = self.img_enc_dec.enc,
      img_dec = self.img_enc_dec.dec,
      text_enc = self.text_enc_dec.enc,
      text_dec = self.text_enc_dec.dec,
    )

  #TODO also roundtrip starting from random emb
  # Single-stage roundtrips:
  # txt -> emb -> txt
  # img -> emb -> img
  # emb -> txt -> emb
  # emb -> img -> emb
  #
  # Two-stage roundtrips:
  # txt -> emb -> img -> emb -> txt
  # img -> emb -> txt -> emb -> img
  def call(self, inputs: tuple[tuple[tf.Tensor, tf.Tensor], tuple[tf.Tensor, tf.Tensor]]):
    img_inputs0, text_inputs0, img_inputs1, text_inputs1 = inputs

    img0_x, img0_y = img0_inputs
    img1_x, img1_y = img1_inputs
    text0_x, text0_y = text0_inputs
    text1_x, text1_y = text1_inputs

    img0_pred = self.img_enc_dec(img0_x)
    text0_pred = self.text_enc_dec(text0_x)

    img1_pred = self.img_rountrip(img1_x)
    text1_pred = self.text_roundtrip(text1_x)

    return img0_pred, text0_pred, img1_pred, text1_pred
    

  
@tf.function
def train_step(real_images):
    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Decode them to fake images
    generated_images = generator(random_latent_vectors)
    # Combine them with real images
    combined_images = tf.concat([generated_images, real_images], axis=0)

    # Assemble labels discriminating real from fake images
    labels = tf.concat(
        [tf.ones((batch_size, 1)), tf.zeros((real_images.shape[0], 1))], axis=0
    )
    # Add random noise to the labels - important trick!
    labels += 0.05 * tf.random.uniform(labels.shape)

    # Train the discriminator
    with tf.GradientTape() as tape:
        predictions = discriminator(combined_images)
        d_loss = loss_fn(labels, predictions)
    grads = tape.gradient(d_loss, discriminator.trainable_weights)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_weights))

    # Sample random points in the latent space
    random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
    # Assemble labels that say "all real images"
    misleading_labels = tf.zeros((batch_size, 1))

    # Train the generator (note that we should *not* update the weights
    # of the discriminator)!
    with tf.GradientTape() as tape:
        predictions = discriminator(generator(random_latent_vectors))
        g_loss = loss_fn(misleading_labels, predictions)
    grads = tape.gradient(g_loss, generator.trainable_weights)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_weights))
    return d_loss, g_loss, generated_images

# def train():
#   import time

#   epochs = 2
#   for epoch in range(epochs):
#       print("\nStart of epoch %d" % (epoch,))
#       start_time = time.time()

#       # Iterate over the batches of the dataset.
#       for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
#           loss_value = train_step(x_batch_train, y_batch_train)

#           # Log every 200 batches.
#           if step % 200 == 0:
#               print(
#                   "Training loss (for one batch) at step %d: %.4f"
#                   % (step, float(loss_value))
#               )
#               print("Seen so far: %d samples" % ((step + 1) * batch_size))

#       # Display metrics at the end of each epoch.
#       train_acc = train_acc_metric.result()
#       print("Training acc over epoch: %.4f" % (float(train_acc),))

#       # Reset training metrics at the end of each epoch
#       train_acc_metric.reset_states()

#       # Run a validation loop at the end of each epoch.
#       for x_batch_val, y_batch_val in val_dataset:
#           test_step(x_batch_val, y_batch_val)

#       val_acc = val_acc_metric.result()
#       val_acc_metric.reset_states()
#       print("Validation acc: %.4f" % (float(val_acc),))
#       print("Time taken: %.2fs" % (time.time() - start_time))

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

  # img0_train, img0_valid = load_img_datasets()
  # text0_train, text0_valid, grapheme_count = load_text_datasets()
  # img1_train, img1_valid = load_img_datasets()
  # text1_train, text1_valid, grapheme_count = load_text_datasets()

  # train = tf.data.Dataset.zip(img0_train, text0_train, img1_train, text1_train)
  # valid = tf.data.Dataset.zip(img0_valid, text0_valid, img1_valid, text1_valid)

  num_heads = 8
  # m = EmbeddingRoundTripper(num_heads=num_heads, vocab_size=grapheme_count)
  emb = 128


  # m.compile(
  #  loss='categorical_crossentropy',
  #  optimizer='adam',
  #  run_eagerly=True,
  # )

  # m.fit(
  #  train,
  #  epochs=20,
  #  validation_data=valid
  # )

  opt = Adam()
  img_enc = ImgEncoder(emb=emb)
  img_dec = ImgDecoder(emb=emb, w=W, h=H)
  txt_enc = TextEncoder(num_heads=num_heads, input_vocab_size=grapheme_count, emb=emb)
  txt_dec = TextDecoder(num_heads=num_heads, emb=emb, output_vocab_size=grapheme_count)

  emb_dataset_ = emb_dataset(batch=BATCH, emb=emb)

 # opt: tf.keras.optimizers.Optimizer,
 # epochs: int,
 # steps_per_epoch: int,
 # img_enc: ImgEncoder,# imb -> emb
 # img_dec: ImgDecoder,# emb -> img
 # txt_enc: TextEncoder,# txt -> emb
 # txt_dec: TextDecoder,# emb -> txt
 # img_bin_op: ImgBinOp,# img ⊕  img -> img 
 # txt_bin_op: TextBinOp,# txt ⊞ txt -> txt
 # emb_dataset: tf.data.Dataset,# infinitely looped, unabeled
 # img_train: tf.data.Dataset,# infinitely looped, unlabeled
 # txt_train: tf.data.Dataset,# infinitely looped, unlabeled
 # img_valid: tf.data.Dataset,# infinitely looped, unlabeled
 # txt_valid: tf.data.Dataset,# infinitely looped, unlabeled
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
