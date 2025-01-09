from collections import Counter
import os

import numpy as np
import tensorflow as tf
import tensorflow_text as tft


from grapheme_idx import GraphemeIdx, load_grapheme_idx
from wptitles import WikipediaTitlesDataset, wp_titles_dataset

BATCH=10
GRAPHEME_QUERY=16
EMBED = 64

def flatten(xss):
 return [x for xs in xss for x in xs]

def pad(a: list[int], pad: int, pad_to: int) -> list[int]:
 print(f"a: {a}")
 print(f"pad_to: {pad_to}")
 return a + [pad] * (pad_to- len(a))

# def index_batch(grapheme_idx: T2I, batched_token_graphemes: list[list[str]]):
#  pad_idx = grapheme_idx.index(grapheme_idx.pad_token)[0]
#  print(f"pad: {pad_idx}")
#  print(f"batched_token_graphemes: {batched_token_graphemes}")

#  max_len = max([len(t) for t in batched_token_graphemes])
#  indices = [flatten(grapheme_idx.index(graphemes)) for graphemes in batched_token_graphemes]
#  print(f"indices: {indices}")
#  # indices = [x[0] for x in indices]
#  # print(f"indices2: {indices}")
#  padded = [pad(token_grapheme_indices, pad_idx, max_len) for token_grapheme_indices in indices]
#  print(f"padded: {padded}")
#  return padded, max_len

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

class BaseAttention(tf.keras.layers.Layer):
 def __init__(self, **kwargs):
  super().__init__()
  self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
  self.layernorm = tf.keras.layers.LayerNormalization()
  self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention):
 def call(self, x, context):
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
 def call(self, x):
  attn_output = self.mha(
      query=x,
      value=x,
      key=x)
  x = self.add([x, attn_output])
  x = self.layernorm(x)
  return x

class CausalSelfAttention(BaseAttention):
 def call(self, x):
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

 def call(self, x):
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

 def call(self, x):
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

 def call(self, x, context):
  x = self.causal_self_attention(x=x)
  x = self.cross_attention(x=x, context=context)

  # Cache the last attention scores for plotting later
  self.last_attn_scores = self.cross_attention.last_attn_scores

  x = self.ffn(x)  # Shape `(batch_size, seq_len, d_model)`.
  return x 

class DecoderOld(tf.keras.layers.Layer):
  @classmethod
  def add_method(cls, fun):
    setattr(cls, fun.__name__, fun)
    return fun

  def __init__(self, grapheme_count: int, units: int):
    # super(Decoder, self).__init__()
    # self.text_processor = text_processor
    # self.vocab_size = text_processor.vocabulary_size()
    # self.word_to_id = tf.keras.layers.StringLookup(
    #     vocabulary=text_processor.get_vocabulary(),
    #     mask_token='', oov_token='[UNK]')
    # self.id_to_word = tf.keras.layers.StringLookup(
    #     vocabulary=text_processor.get_vocabulary(),
    #     mask_token='', oov_token='[UNK]',
    #     invert=True)
    # self.start_token = self.word_to_id('[START]')
    # self.end_token = self.word_to_id('[END]')

    # self.units = units


    # # 1. The embedding layer converts token IDs to vectors
    # self.embedding = tf.keras.layers.Embedding(self.vocab_size,
    #                                            units, mask_zero=True)

    # 2. The RNN keeps track of what's been generated so far.
    self.rnn = tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

    # 3. The RNN output will be the query for the attention layer.
    self.attention = CrossAttention(units)

    # 4. This fully connected layer produces the logits for each
    # output grapheme.
    self.output_layer = tf.keras.layers.Dense(grapheme_count)

    @Decoder.add_method
    def call(self,
             token_emb,
             state=None,
             return_state=False):  
      # shape_checker = ShapeChecker()
      # shape_checker(x, 'batch t')
      # shape_checker(context, 'batch s units')

      # # 1. Lookup the embeddings
      # x = self.embedding(x)
      # shape_checker(x, 'batch t units')

      # 2. Process the target sequence.
      x, state = self.rnn(token_emb)
      shape_checker(x, 'batch t units')

      # 3. Use the RNN output as the query for the attention over the context.
      x = self.attention(x, context)
      self.last_attention_weights = self.attention.last_attention_weights
      # shape_checker(x, 'batch t units')
      # shape_checker(self.last_attention_weights, 'batch t s')

      # Step 4. Generate logit predictions for the next token.
      logits = self.output_layer(x)
      shape_checker(logits, 'batch t target_vocab_size')

      if return_state:
        return logits, state
      else:
        return logits

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
  print(f"Decoder x.get_shape: {x.get_shape()}")
  print(f"Decoder context.get_shape: {context.get_shape()}")
  # `x` is token-IDs shape (batch, target_seq_len)
  x = self.pos_embedding(x)  # (batch_size, target_seq_len, d_model)

  x = self.dropout(x)

  for i in range(self.num_layers):
    x  = self.dec_layers[i](x, context)

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

 def call(self, inputs):
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


# class WordAutoencoder(tf.keras.layers.Layer):
#     def __init__(self, grapheme_count: int):
#         super().__init__()

#         self.transformer = Transformer(num_layers, d_model: int, num_heads: int, dff: int,
#          input_vocab_size: int, target_vocab_size: int, dropout_rate=0.1
#         )

#         # self.grapheme_enc_query = self.add_weight(shape=(BATCH, GRAPHEME_QUERY, EMBED), initializer="random_normal", trainable=True)
#         # self.grapheme_dec_query = self.add_weight(shape=(BATCH, GRAPHEME_QUERY, grapheme_count), initializer="random_normal", trainable=True)
#         self.grapheme_emb = tf.keras.layers.Embedding(
#          grapheme_count,
#          EMBED,
#         )
#         # self.grapheme_attn_enc = tf.keras.layers.Attention()
#         # self.grapheme_attn_dec = tf.keras.layers.Attention()
#         # self.grapheme_attn = tf.keras.layers.MultiHeadAttention(key_dim=EMBED, num_heads=1)

#     # token_grapheme_indices: (batch, token, grapheme); integers representing token indices 
#     def call(self, token_grapheme_indices: tf.Tensor):
#      print(grapheme_indices)
#      # shape: batch, token, grapheme

#      grapheme_embs = self.grapheme_emb(grapheme_indices)
#      print(grapheme_embs)

#      # token_emb = self.grapheme_attn([self.grapheme_query, grapheme_embs])
#      # token_emb = self.grapheme_attn(query=self.grapheme_query, value=grapheme_embs)

#      #TODO decoder

#      # prior = tf.zeros((BATCH, ))
#      # while True:

#      output = self.transformer(grapheme_embs)

#      return token_emb

class Translator(tf.Module):
  def __init__(self, grapheme_idx: GraphemeIdx, transformer: Transformer):
   self.grapheme_idx = grapheme_idx
   self.transformer = transformer

  # batch_size: the length of the 
  # tokens: list of strings of shape (batch,)
  def __call__(self, tokens: tf.RaggedTensor, int_dtype=tf.int64):
    print(f"Translator tokens: {tokens}")
    print(f"Translator tokens type: {type(tokens)}")

    print(f"Translator eager? {tf.executing_eagerly()}")
    # grapheme_indices = self.grapheme_idx.index_tokens(tokens)
    # max_len = max([len(t) for t in grapheme_indices])
    # grapheme_indices = tf.constant(grapheme_indices, dtype=int_dtype, shape=(len(tokens), max_len))
    # encoder_input = grapheme_indices
    encoder_input = tokens

    batch_size = tokens.get_shape()[0]
    print(f"Translator batch_size: {batch_size}")

    # Initialize the output with a start token, and prep the end token
    starts = tf.constant([self.grapheme_idx.start_idx()] * batch_size, dtype=int_dtype)
    end = tf.constant(self.grapheme_idx.end_idx(), dtype=int_dtype)

    # `tf.TensorArray` is required here (instead of a Python list), so that the
    # dynamic-loop can be traced by `tf.function`.
    output_array = tf.TensorArray(dtype=int_dtype, size=0, dynamic_size=True)
    output_array = output_array.write(0, starts)

    for i in tf.range(MAX_STR_LEN):
      output = tf.transpose(output_array.stack())
      print(f"Translator encoder_input.get_shape(): {encoder_input.get_shape()}")
      print(f"Translator output.get_shape(): {output.get_shape()}")
      predictions = self.transformer([encoder_input, output], training=False)

      print(f"Translator predictions.get_shape: {predictions.get_shape()}")

      # Select the last token from the `seq_len` dimension.
      predictions = predictions[:, -1:, :]  # Shape `(batch_size, 1, vocab_size)`.

      predictions = tf.squeeze(predictions, axis=1)

      print(f"Translator predictions.get_shape 2: {predictions.get_shape()}")

      predicted_id = tf.argmax(predictions, axis=-1)

      print(f"Translator predicted_id.get_shape: {predicted_id.get_shape()}")

      # Concatenate the `predicted_id` to the output which is given to the
      # decoder as its input.
      output_array = output_array.write(i+1, predicted_id)

      # ended = predicted_id == end

      # print(f"Translator ended.get_shape: {ended.get_shape()}")
      # print(f"Translator ended.dtype: {ended.dtype}")


      # if ended.all():
      #   break

      #TODO early exit if all outputs have emitted an end token

    output = tf.transpose(output_array.stack())

    print(f"Translator output.get_shape: {output.get_shape()}")

    # The output shape is `(1, tokens)`.
    # text = tokenizers.en.detokenize(output)[0]  # Shape: `()`.
    reconstructed_tokens: list[str] = self.grapheme_idx.unindex_tokens(output.numpy())

    # tokens = tokenizers.en.lookup(output)[0]

    # `tf.function` prevents us from using the attention_weights that were
    # calculated on the last iteration of the loop.
    # So, recalculate them outside the loop.
    self.transformer([encoder_input, output[:,:-1]], training=False)
    attention_weights = self.transformer.decoder.last_attn_scores

    return reconstructed_tokens, tokens, attention_weights

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

  def call(self, x):
   return self.translator(x)

def titles_datum_extractor(title, tokenizer):
 print(f"titles_datum_extractor title {title}")
 tokens = tokenizer.tokenize(title)
 print(f"titles_datum_extractor tokens {tokens}")
 return (tokens, tokens)# return as x and y as this is an autoencoder

# def tokenize_str(s: tf.Tensor) -> tf.RaggedTensor:
#  print(s.numpy())
#  # FIXME
#  return s

# titles shape: (batch,None,)
# def tokenize(titles: tf.RaggedTensor) -> tf.RaggedTensor:
#  print(f"titles shape: {titles.shape}")
#  print(f"titles type: {type(titles)}")
#  return tf.map_fn(tokenize_str, titles)

MAX_STR_LEN = 16

if __name__=="__main__":
 with tf.device('/CPU:0'):
  print(f"Eager? {tf.executing_eagerly()}")
  home_dir = os.path.expanduser('~')
  text_dir = os.path.join(home_dir, 'Data', 'org', 'gutenberg', 'mirror_txt')
  # img_dir = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'wikimedia-commons-hires-png_not-too-big')

  grapheme_idx = load_grapheme_idx()
  print(grapheme_idx)

  

  # tokenizer = tft.UnicodeScriptTokenizer()
  tokenizer = tft.WhitespaceTokenizer()
  def tokenize(s: str):
   return tokenizer.tokenize(s)
  
 #  # text = GutenbergTextDataset(text_dir)
 #  # text = text.map(lambda x: tft.ngrams(tokenizer.tokenize(x), 5, reduction_type=tft.Reduction.STRING_JOIN, string_separator='\x00'))
 #  # text = text.shuffle(200)

  wptitles_train_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_train.gz')
  wptitles_valid_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_valid.gz')
  # wptitles_test_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0_test.gz')

  # train = tf.data.TextLineDataset(wptitles_train_path, compression_type='GZIP')
  # valid = tf.data.TextLineDataset(wptitles_valid_path, compression_type='GZIP')

  # train = WikipediaTitlesDataset(wptitles_train_path)
  # valid = WikipediaTitlesDataset(wptitles_valid_path)
  # test= WikipediaTitlesDataset(wptitles_test_path).map(lambda x: tokenizer.tokenize(x))

  train = wp_titles_dataset(wptitles_train_path, tokenizer, grapheme_idx, MAX_STR_LEN)
  valid = wp_titles_dataset(wptitles_valid_path, tokenizer, grapheme_idx, MAX_STR_LEN)

  # train_gen = gen_wp_titles_dataset(wptitles_train_path, tokenizer, grapheme_idx)
  # valid_gen = gen_wp_titles_dataset(wptitles_valid_path, tokenizer, grapheme_idx)


  # Tokenize
  # train = train.map(tokenize)
  # valid = valid.map(tokenize)

  train_iter = iter(train)
  valid_iter = iter(valid)
  print(f"First train: {next(train_iter)}")
  print(f"First valid: {next(valid_iter)}")

  train = train.ragged_batch(BATCH, drop_remainder=True)
  valid = valid.ragged_batch(BATCH, drop_remainder=True)
  # test = test.batch(BATCH)

  train_iter = iter(train)
  valid_iter = iter(valid)
  print(f"First train batch: {next(train_iter)}")
  print(f"First valid batch: {next(valid_iter)}")

  num_layers = 4
  d_model = 128
  dff = 512
  num_heads = 8
  dropout_rate = 0.1

  grapheme_count = len(grapheme_idx)
  print(f"grapheme_count: {grapheme_count}")

 #  transformer = Transformer(
 #    num_layers=num_layers,
 #    d_model=d_model,
 #    num_heads=num_heads,
 #    dff=dff,
 #    input_vocab_size=grapheme_count ,
 #    target_vocab_size=grapheme_count ,
 #    dropout_rate=dropout_rate
 #  )

 #  # m = WordAutoencoder(grapheme_idx.t2i.highest_idx + 1)
 # # model.compile(optimizer='adam',
 # #   loss='sparse_categorical_crossentropy',
 # #   metrics=['accuracy'])
 # # model.fit(x_train, y_train, epochs=5)
 # # model.evaluate(x_test, y_test)

 #  m = Translator(grapheme_idx, transformer)
  # def __init__(self, num_layers: int, d_model: int, num_heads: int, dff: int, grapheme_count: int, dropout_rate: float, **kwargs):
  m = WordAutoencoderModel(num_layers, d_model, num_heads, dff, grapheme_count, dropout_rate)

  m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'], run_eagerly=True)
  m.fit(train, epochs=10)
  # m.evaluate(valid)



  # it_txt = iter(text)
  # available_tokens: list[str] = list()
  # while True:
  #  try:
  #   txt = next(it_txt)
  #   tokens = txt.to_list()[0]
  #   tokens = [t.decode() for t in tokens]

  #   available_tokens.extend(tokens)

  #   while len(available_tokens) >= BATCH:
  #    batch = available_tokens[:BATCH]
  #    available_tokens = available_tokens[BATCH:]

  #    print(f"batch len: {len(batch)}")
  #    print(f"available tokens: {len(available_tokens)}")


  #    output, _, __ = m(batch)
  #    print(output)
  #    batch = list()

  #  except StopIteration:
  #   pass
