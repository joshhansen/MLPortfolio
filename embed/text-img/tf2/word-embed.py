from collections import Counter
import os

import numpy as np
import tensorflow as tf
import tensorflow_text as tft


from grapheme_idx import GraphemeIdx, load_grapheme_idx
from wptitles import wp_titles_dataset

BATCH=10
GRAPHEME_QUERY=16
EMBED = 64


def positional_encoding(length, depth):
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
 def __init__(self, vocab_size, d_model):
  super().__init__()
  self.d_model = d_model
  self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True) 
  self.pos_encoding = positional_encoding(length=2048, depth=d_model)

 def compute_mask(self, *args, **kwargs):
  return self.embedding.compute_mask(*args, **kwargs)

 def call(self, x: tf.Tensor):
  # print(f"x.get_shape: {x.get_shape()}")
  length = x.get_shape()[1]
  # print(f"x.get_shape()[1]: {x.get_shape()[1]}")
  x = self.embedding(x)
  # This factor sets the relative scale of the embedding and positonal_encoding.
  x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
  x = x + self.pos_encoding[tf.newaxis, :length, :]
  return x

# multi-head attention + layernorm + residual connection
class BaseAttention(tf.keras.layers.Layer):
 def __init__(self, **kwargs):
  super().__init__()
  self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
  self.layernorm = tf.keras.layers.LayerNormalization()
  self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
 def call(self, x: tf.Tensor, context: tf.Tensor):
  attn_output, attn_scores = self.mha(
      query=x,
      key=context,
      value=context,
      return_attention_scores=True)

  # Cache the attention scores for plotting later.
  self.last_attn_scores = attn_scores

  x = self.add([x, attn_output])
  x = self.layernorm(x)

  return x

class GlobalSelfAttention(BaseAttention):
 def call(self, x: tf.Tensor):
  attn_output = self.mha(
      query=x,
      value=x,
      key=x)
  x = self.add([x, attn_output])
  x = self.layernorm(x)
  return x

class CausalSelfAttention(BaseAttention):
 def call(self, x: tf.Tensor):
  attn_output = self.mha(
      query=x,
      value=x,
      key=x,
      use_causal_mask = True)
  x = self.add([x, attn_output])
  x = self.layernorm(x)
  return x

class FeedForward(tf.keras.layers.Layer):
 def __init__(self, d_model: int, dff: int, dropout_rate=0.1):
  super().__init__()
  self.seq = tf.keras.Sequential([
    tf.keras.layers.Dense(dff, activation='relu'),
    tf.keras.layers.Dense(d_model),
    tf.keras.layers.Dropout(dropout_rate)
  ])
  self.add = tf.keras.layers.Add()
  self.layer_norm = tf.keras.layers.LayerNormalization()

 def call(self, x: tf.Tensor):
  x = self.add([x, self.seq(x)])
  x = self.layer_norm(x) 
  return x

class EncoderLayer(tf.keras.layers.Layer):
 def __init__(self,*, d_model: int, num_heads: int, dff: int, dropout_rate=0.1):
  super().__init__()

  self.self_attention = GlobalSelfAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate)

  self.ffn = FeedForward(d_model, dff)

 def call(self, x: tf.Tensor):
  x = self.self_attention(x)
  x = self.ffn(x)
  return x

class Encoder(tf.keras.layers.Layer):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int,
             dff: int, vocab_size: int, dropout_rate=0.1):
  super().__init__()

  self.d_model = d_model
  self.num_layers = num_layers

  self.pos_embedding = PositionalEmbedding(
      vocab_size=vocab_size, d_model=d_model)

  self.enc_layers = [
      EncoderLayer(d_model=d_model,
                   num_heads=num_heads,
                   dff=dff,
                   dropout_rate=dropout_rate)
      for _ in range(num_layers)]
  self.dropout = tf.keras.layers.Dropout(dropout_rate)

 # x: (batch, seq_len); token IDs (int datatype)
 def call(self, x: tf.Tensor):
  x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

  # Add dropout.
  x = self.dropout(x)

  for i in range(self.num_layers):
    x = self.enc_layers[i](x)

  return x  # Shape `(batch_size, seq_len, d_model)`.


class DecoderLayer(tf.keras.layers.Layer):
 def __init__(self,
             *,
             d_model: int,
             num_heads: int,
             dff: int,
             dropout_rate=0.1):
  super(DecoderLayer, self).__init__()

  self.causal_self_attention = CausalSelfAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate)

  self.cross_attention = CrossAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate)

  self.ffn = FeedForward(d_model, dff)

 def call(self, x: tf.Tensor, context: tf.Tensor):
  x = self.causal_self_attention(x=x)
  x = self.cross_attention(x=x, context=context)

  # Cache the last attention scores for plotting later
  self.last_attn_scores = self.cross_attention.last_attn_scores

  x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
  return x

 
class DecoderLayerNoContext(tf.keras.layers.Layer):
 def __init__(self,
             *,
             d_model: int,
             num_heads: int,
             dff: int,
             dropout_rate=0.1):
  super(DecoderLayerNoContext, self).__init__()

  self.causal_self_attention = CausalSelfAttention(
      num_heads=num_heads,
      key_dim=d_model,
      dropout=dropout_rate)

  self.ffn = FeedForward(d_model, dff)

 def call(self, x: tf.Tensor):
  x = self.causal_self_attention(x=x)

  x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
  return x 

class Decoder(tf.keras.layers.Layer):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int,
              vocab_size: int, dropout_rate=0.1):
  super(Decoder, self).__init__()

  self.d_model = d_model
  self.num_layers = num_layers

  self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                           d_model=d_model)
  self.dropout = tf.keras.layers.Dropout(dropout_rate)
  self.dec_layers = [
      DecoderLayer(d_model=d_model, num_heads=num_heads,
                   dff=dff, dropout_rate=dropout_rate)
      for _ in range(num_layers)]

  self.last_attn_scores = None

 def call(self, x: tf.Tensor, context: tf.Tensor):
  # print(f"Decoder x.get_shape: {x.get_shape()}")
  # print(f"Decoder context.get_shape: {context.get_shape()}")
  # `x` is token-IDs shape (batch, target_seq_len)
  x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

  x = self.dropout(x)

  for i in range(self.num_layers):
    x  = self.dec_layers[i](x, context)

  self.last_attn_scores = self.dec_layers[-1].last_attn_scores

  # The shape of x is (batch_size, target_seq_len, d_model).
  return x

class DecoderNoContext(tf.keras.Model):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int,
              vocab_size: int, dropout_rate=0.1):
  super(DecoderNoContext, self).__init__()

  self.d_model = d_model
  self.num_layers = num_layers

  self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,
                                           d_model=d_model)
  self.dropout = tf.keras.layers.Dropout(dropout_rate)
  self.dec_layers = [
      DecoderLayerNoContext(d_model=d_model, num_heads=num_heads,
                   dff=dff, dropout_rate=dropout_rate)
      for _ in range(num_layers)]

  self.last_attn_scores = None

 def call(self, x: tf.Tensor):
  # print(f"Decoder x.get_shape: {x.get_shape()}")
  # print(f"Decoder context.get_shape: {context.get_shape()}")
  # `x` is token-IDs shape (batch, target_seq_len)
  x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

  x = self.dropout(x)

  for i in range(self.num_layers):
    x  = self.dec_layers[i](x)

  self.last_attn_scores = self.dec_layers[-1].last_attn_scores

  # The shape of x is (batch_size, target_seq_len, d_model).
  return x
 

class Transformer(tf.keras.Model):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int,
             input_vocab_size: int, target_vocab_size: int, dropout_rate=0.1):
  super().__init__()
  self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                         num_heads=num_heads, dff=dff,
                         vocab_size=input_vocab_size,
                         dropout_rate=dropout_rate)

  self.decoder = Decoder(num_layers=num_layers, d_model=d_model,
                         num_heads=num_heads, dff=dff,
                         vocab_size=target_vocab_size,
                         dropout_rate=dropout_rate)

  self.final_layer = tf.keras.layers.Dense(target_vocab_size)

 def call(self, inputs: tuple[tf.Tensor, tf.Tensor], *args, **kwargs):
  print(f"Transformer inputs: {inputs}")
  # print(f"Transformer inputs shape: {inputs.get_shape()}")
  # print(f"Transformer args: {args}")
  # print(f"Transformer kwargs: {kwargs}")
  # To use a Keras model with `.fit` you must pass all your inputs in the
  # first argument.
  context, x  = inputs

  print(f"Transformer x.get_shape: {x.get_shape()}")
  print(f"Transformer context.get_shape: {context.get_shape()}")

  context = self.encoder(context)  # (batch_size, context_len, d_model)

  print(f"Transformer context.get_shape 2: {context.get_shape()}")

  x = self.decoder(x, context)  # (batch_size, target_len, d_model)

  print(f"Transformer x.get_shape 2: {x.get_shape()}")

  # Final linear layer output.
  logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

  print(f"Transformer logits.get_shape: {logits.get_shape()}")

  try:
    # Drop the keras mask, so it doesn't scale the losses/metrics.
    # b/250038731
    del logits._keras_mask
  except AttributeError:
    pass

  # Return the final output and the attention weights.
  return logits

class Text2Emb(tf.keras.Model):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int,
             input_vocab_size: int, dropout_rate=0.1):
  super(Text2Emb, self).__init__()
  self.encoder = Encoder(num_layers=num_layers, d_model=d_model,
                         num_heads=num_heads, dff=dff,
                         vocab_size=input_vocab_size,
                         dropout_rate=dropout_rate)



 def call(self, input: tf.Tensor, *args, **kwargs):
  print(f"Text2Emb input: {input}")
  # (batch, seq)

  seq_with_attn = self.encoder(input)

  # (batch, seq, attn)

  out = tf.reduce_sum(seq_with_attn, 1)

  # (batch, attn)

  return out

class Emb2Text(tf.keras.Model):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int,
             target_vocab_size: int, dropout_rate=0.1):
  super(Emb2Text, self).__init__()

  self.decoder = DecoderNoContext(
   num_layers=num_layers,
   d_model=d_model,
   num_heads=num_heads,
   dff=dff,
   vocab_size=target_vocab_size,
   dropout_rate=dropout_rate
  )
  self.final_layer = tf.keras.layers.Dense(target_vocab_size)

 def call(self, input: tf.Tensor, *args, **kwargs):
  print(f"EmbToText input: {input}")

  # (batch, emb)

  # Copy the embedding into every potential sequence location
  expanded = tf.repeat(input, MAX_STR_LEN, 1)
  # (batch, seq, emb)

  x = self.decoder(expanded)  # (batch_size, target_len, d_model)

  # Final linear layer output.
  logits = self.final_layer(x)  # (batch_size, target_len, target_vocab_size)

  try:
    # Drop the keras mask, so it doesn't scale the losses/metrics.
    # b/250038731
    del logits._keras_mask
  except AttributeError:
    pass

  # Return the final output and the attention weights.
  return logits

class Text2Emb2Text(tf.keras.Model):
 def __init__(self, *, num_layers: int, d_model: int, num_heads: int, dff: int,
             input_vocab_size: int, target_vocab_size: int, dropout_rate=0.1):
  super().__init__()
  self.enc = Text2Emb(
   num_layers=num_layers,
   d_model=d_model,
   num_heads=num_heads,
   dff=dff,
   input_vocab_size=input_vocab_size,
   dropout_rate=dropout_rate
  )
  self.dec = Emb2Text(
   num_layers=num_layers,
   d_model=d_model,
   num_heads=num_heads,
   dff=dff,
   target_vocab_size=target_vocab_size,
   dropout_rate=dropout_rate
  )

 def call(self, inputs: tuple[tf.Tensor, tf.Tensor]):
  x, y = inputs
  return self.dec(self.enc(x))
 

class Translator(tf.Module):
  def __init__(self, grapheme_idx: GraphemeIdx, transformer: Transformer):
   self.grapheme_idx = grapheme_idx
   self.transformer = transformer

  # tokens: shape (batch, grapheme)
  def __call__(self, tokens: tf.Tensor, int_dtype=tf.int64):
    # print(f"Translator tokens shape: {tokens.shape}")
    # print(f"Translator tokens: {tokens}")
    # print(f"Translator tokens type: {type(tokens)}")
    # print(f"Translator eager? {tf.executing_eagerly()}")
    encoder_input = tokens

    batch_size = tokens.get_shape()[0]
    grapheme_count = len(self.grapheme_idx)

    # Initialize the output with a start token, and prep the end token
    starts = tf.constant([self.grapheme_idx.start_idx()] * batch_size, dtype=int_dtype)

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=int_dtype, size=0, dynamic_size=True)
    output_array = output_array.write(0, starts)

    probs_array = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)

    # insert one-hot probabilities predicting the start token, which is given
    # In other words, the loss function wants the matrix of probabilities considered while decoding
    # The start token is given a probability of 1.0 since we must "decode" it as the first step
    # So we one-hot-encode the start token index interpret it as probabilities
    # start_probs shape: (batch_size, grapheme_count)
    start_probs = tf.one_hot([self.grapheme_idx.start_idx()] * batch_size, grapheme_count)
    probs_array = probs_array.write(0, start_probs)

    for i in tf.range(MAX_STR_LEN+1):
      output = tf.transpose(output_array.stack())
      # print(f"Translator encoder_input.get_shape(): {encoder_input.get_shape()}")
      # print(f"Translator output.get_shape(): {output.get_shape()}")
      predictions = self.transformer([encoder_input, output], training=False)

      # print(f"Translator predictions.get_shape: {predictions.get_shape()}")

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predictions = tf.squeeze(predictions, axis=1)

      # print(f"Translator predictions.get_shape 2: {predictions.get_shape()}")
      probs_array = probs_array.write(i+1, predictions)

      predicted_id = tf.argmax(predictions, axis=-1)

      # print(f"Translator predicted_id.get_shape: {predicted_id.get_shape()}")

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id)

    output = tf.transpose(output_array.stack())

    # print(f"Translator output.get_shape: {output.get_shape()}")

    # The output shape is `(1, tokens)`.
    # text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.
    reconstructed_tokens: list[str] = self.grapheme_idx.unindex_tokens(output.numpy())

    # tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    
    probs = probs_array.stack()
    # print(f"probs shape: {probs.get_shape()}")

    probs = tf.transpose(probs, [1, 0, 2])
    # print(f"probs shape 2: {probs.get_shape()}")

    return reconstructed_tokens, tokens, attention_weights, probs

# @tf.keras.saving.register_keras_serializable()
class WordAutoencoderModel(tf.keras.Model):
  def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, grapheme_count: int, dropout_rate: float, **kwargs):
    super().__init__(**kwargs)

    transformer = Transformer(
      num_layers=num_layers,
      d_model=d_model,
      num_heads=num_heads,
      dff=dff,
      input_vocab_size=grapheme_count,
      target_vocab_size=grapheme_count,
      dropout_rate=dropout_rate
    )

    self.translator = Translator(grapheme_idx, transformer)

  def call(self, x: tf.Tensor):
   print(f"WEM x shape: {x.get_shape()}")
   tokens_as_text = self.translator.grapheme_idx.unindex_tokens(x.numpy())
   reconstructed_tokens, tokens, attention_weights, probs = self.translator(x)

   print("tokens_as_text:")
   for t in tokens_as_text:
    print(f"\t{t}")

   print("reconstructed_tokens:")
   for t in reconstructed_tokens:
    print(f"\t{t}")

   return probs

MAX_STR_LEN = 32

def make_masked_loss(start_idx: int):
 def masked_loss(label: tf.Tensor, pred: tf.Tensor):
   mask = label != 0
   loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
     from_logits=True,
     reduction='none'
   )

   # # Re-add the start token
   # batch_size = label.get_shape()[0]
   # starts = tf.constant([[start_idx]] * batch_size, dtype=label.dtype)
   # label = tf.concat([starts, label], 1)

   pred = pred[:, 1:]# drop the start token
  
   loss = loss_object(label, pred)

   mask = tf.cast(mask, dtype=loss.dtype)
   loss *= mask

   loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)
   return loss
 return masked_loss


def masked_accuracy(label, pred):
  pred = tf.argmax(pred, axis=2)
  label = tf.cast(label, pred.dtype)
  match = label == pred

  mask = label != 0

  match = match & mask

  match = tf.cast(match, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(match)/tf.reduce_sum(mask)


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super().__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    step = tf.cast(step, dtype=tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# MAX_TOKENS=128
def prepare_batch(x: tf.Tensor, y: tf.Tensor):

    # pt = tokenizers.pt.tokenize(pt)      # Output is ragged.
    # pt = pt[:, :MAX_TOKENS]    # Trim to MAX_TOKENS.
    # pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    # en = tokenizers.en.tokenize(en)
    # en = en[:, :(MAX_TOKENS+1)]
    # en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    # en_labels = en[:, 1:].to_tensor()   # Drop the [START] tokens

    labels = y[:, 1:]

    return (x, y), labels

if __name__=="__main__":
 # with tf.device('/CPU:0'):
 with tf.device('gpu'):
  # print(f"Eager? {tf.executing_eagerly()}")
  home_dir = os.path.expanduser('~')
  text_dir = os.path.join(home_dir, 'Data', 'org', 'gutenberg', 'mirror_txt')
  # img_dir = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'wikimedia-commons-hires-png_not-too-big')

  grapheme_idx = load_grapheme_idx()
  print(grapheme_idx)

  # tokenizer = tft.UnicodeScriptTokenizer()
  tokenizer = tft.WhitespaceTokenizer()
  def tokenize(s: str):
   return tokenizer.tokenize(s)
  
  wptitles_train_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_train.gz')
  wptitles_valid_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_valid.gz')
  # wptitles_test_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_test.gz')

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

  print(f"First train batch: {next(iter(train))}")
  print(f"First valid batch: {next(iter(valid))}")

  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1

  grapheme_count = len(grapheme_idx)
  print(f"grapheme_count: {grapheme_count}")

  # def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, grapheme_count: int, dropout_rate: float, **kwargs):
  # m = WordAutoencoderModel(num_layers, d_model, num_heads, dff, grapheme_count, dropout_rate)

  # m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
  # m.fit(train, epochs=10)
  # m.evaluate(valid)

  learning_rate = CustomSchedule(d_model)

  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

  # transformer = Transformer(
  #   num_layers=num_layers,
  #   d_model=d_model,
  #   num_heads=num_heads,
  #   dff=dff,
  #   input_vocab_size=grapheme_count,
  #   target_vocab_size=grapheme_count,
  #   dropout_rate=dropout_rate
  # )

  text2emb2text = Text2Emb2Text(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=grapheme_count,
    target_vocab_size=grapheme_count,
    dropout_rate=dropout_rate
  )

  masked_loss = make_masked_loss(grapheme_idx.start_idx())

  text2emb2text.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy],
    run_eagerly=True,
   )

  text2emb2text.fit(train,
                epochs=20,
                validation_data=valid)


