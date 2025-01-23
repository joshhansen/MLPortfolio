from img_enc_dec import Encoder as ImgEncoder, Decoder as ImgDecoder, EncDec as ImgEncDec, load_datasets as load_img_datasets, W, H
from text_enc_dec import Encoder as TextEncoder, Decoder as TextDecoder, EncDec as TextEncDec, MAX_STR_LEN, prepare_batch as prepare_text_batch, load_datasets as load_text_datasets

import tensorflow as tf

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

def train():
  import time

  epochs = 2
  for epoch in range(epochs):
      print("\nStart of epoch %d" % (epoch,))
      start_time = time.time()

      # Iterate over the batches of the dataset.
      for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
          loss_value = train_step(x_batch_train, y_batch_train)

          # Log every 200 batches.
          if step % 200 == 0:
              print(
                  "Training loss (for one batch) at step %d: %.4f"
                  % (step, float(loss_value))
              )
              print("Seen so far: %d samples" % ((step + 1) * batch_size))

      # Display metrics at the end of each epoch.
      train_acc = train_acc_metric.result()
      print("Training acc over epoch: %.4f" % (float(train_acc),))

      # Reset training metrics at the end of each epoch
      train_acc_metric.reset_states()

      # Run a validation loop at the end of each epoch.
      for x_batch_val, y_batch_val in val_dataset:
          test_step(x_batch_val, y_batch_val)

      val_acc = val_acc_metric.result()
      val_acc_metric.reset_states()
      print("Validation acc: %.4f" % (float(val_acc),))
      print("Time taken: %.2fs" % (time.time() - start_time))

if __name__=="__main__":
  img0_train, img0_valid = load_img_datasets()
  text0_train, text0_valid, grapheme_count = load_text_datasets()
  img1_train, img1_valid = load_img_datasets()
  text1_train, text1_valid, grapheme_count = load_text_datasets()

  train = tf.data.Dataset.zip(img0_train, text0_train, img1_train, text1_train)
  valid = tf.data.Dataset.zip(img0_valid, text0_valid, img1_valid, text1_valid)

  num_heads = 8
  m = EmbeddingRoundTripper(num_heads=num_heads, vocab_size=grapheme_count)

  m.compile(
   loss='categorical_crossentropy',
   optimizer='adam',
   run_eagerly=True,
  )

  m.fit(
   train,
   epochs=20,
   validation_data=valid
  )
