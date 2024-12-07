from collections import Counter
import os

import tensorflow as tf
import tensorflow_text as tft

from t2i import T2I

from wptitles import WikipediaTitlesDataset

BATCH=10
GRAPHEME_QUERY=16
EMBED = 64

def flatten(xss):
 return [x for xs in xss for x in xs]

def pad(a: list[int], pad: int, pad_to: int) -> list[int]:
 print(f"a: {a}")
 print(f"pad_to: {pad_to}")
 return a + [pad] * (pad_to- len(a))

def index_batch(grapheme_idx: T2I, batched_token_graphemes: list[list[str]]):
 pad_idx = grapheme_idx.index(grapheme_idx.pad_token)[0]
 print(f"pad: {pad_idx}")
 print(f"batched_token_graphemes: {batched_token_graphemes}")

 max_len = max([len(t) for t in batched_token_graphemes])
 indices = [flatten(grapheme_idx.index(graphemes)) for graphemes in batched_token_graphemes]
 print(f"indices: {indices}")
 # indices = [x[0] for x in indices]
 # print(f"indices2: {indices}")
 padded = [pad(token_grapheme_indices, pad_idx, max_len) for token_grapheme_indices in indices]
 print(f"padded: {padded}")
 return padded, max_len

class WordAutoencoder(tf.keras.layers.Layer):
    def __init__(self, grapheme_count: int):
        super().__init__()

        self.grapheme_query = self.add_weight(shape=(BATCH, GRAPHEME_QUERY, EMBED), initializer="random_normal", trainable=True)
        self.grapheme_emb = tf.keras.layers.Embedding(
         grapheme_count,
         EMBED,
        )
        self.grapheme_attn = tf.keras.layers.Attention()


    # token_grapheme_indices: (batch, token, grapheme); integers representing token indices 
    def call(self, token_grapheme_indices: tf.Tensor):
     print(grapheme_indices)
     # shape: batch, token, grapheme

     grapheme_embs = self.grapheme_emb(grapheme_indices)
     print(grapheme_embs)

     token_emb = self.grapheme_attn([self.grapheme_query, grapheme_embs])

     #TODO decoder

     return token_emb


if __name__=="__main__":
 with tf.device('/CPU:0'):
  home_dir = os.path.expanduser('~')
  text_dir = os.path.join(home_dir, 'Data', 'org', 'gutenberg', 'mirror_txt')
  img_dir = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'wikimedia-commons-hires-png_not-too-big')

  grapheme_counts_path = os.path.join(home_dir, 'Projects', 'ML', 'MLPortfolio', 'embed', 'text-img', 'tf2', 'grapheme_counts_9995.tsv')
  grapheme_idx = dict()
  with open(grapheme_counts_path, 'rt') as r:
   for i, l in enumerate(r):
    g, c, rel_c, cum_rel_c = l.split('\t')
    cum_rel_c = cum_rel_c[:-1]
    c = int(c)
    rel_c = float(rel_c)
    cum_rel_c = float(cum_rel_c)

    grapheme_idx[g] = i

  print(grapheme_idx)

  grapheme_idx = T2I(grapheme_idx)
  print(grapheme_idx)

  

  # tokenizer = tft.UnicodeScriptTokenizer()
  tokenizer = tft.WhitespaceTokenizer()
 #  # text = GutenbergTextDataset(text_dir)
 #  # text = text.map(lambda x: tft.ngrams(tokenizer.tokenize(x), 5, reduction_type=tft.Reduction.STRING_JOIN, string_separator='\x00'))
 #  # text = text.shuffle(200)

  wptitles_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0.gz')
  text = WikipediaTitlesDataset(wptitles_path)
  text = text.map(lambda x: tokenizer.tokenize(x))


  m = WordAutoencoder(grapheme_idx.t2i.highest_idx + 1)
 # model.compile(optimizer='adam',
 #   loss='sparse_categorical_crossentropy',
 #   metrics=['accuracy'])
 # model.fit(x_train, y_train, epochs=5)
 # model.evaluate(x_test, y_test)




  it_txt = iter(text)
  available_tokens = list()
  while True:
   try:
    txt = next(it_txt)
    tokens = txt.to_list()[0]
    tokens = [t.decode() for t in tokens]

    available_tokens.extend(tokens)

    while len(available_tokens) >= BATCH:
     batch = available_tokens[:BATCH]
     available_tokens = available_tokens[BATCH:]

     print(f"batch len: {len(batch)}")
     print(f"available tokens: {len(available_tokens)}")

     graphemes = [list(t) for t in batch]
     print(len(graphemes))
     print(graphemes)

     grapheme_indices, max_len = index_batch(grapheme_idx, graphemes)
     print(grapheme_indices)
     print(len(grapheme_indices))

     grapheme_indices = tf.constant(grapheme_indices, dtype=tf.int32, shape=(BATCH, max_len))

     emb = m(grapheme_indices)
     print(emb)
     batch = list()
    # # tokens = txt.to_list()[0]
    # # graphemes = [list(t.decode()) for t in tokens]
    # graphemes = tf.strings.split(txt, '')

    # print(graphemes)
    # grapheme_indices = grapheme_idx.index(graphemes)
    # print(grapheme_indices)
    # grapheme_indices = tf.ragged.constant(grapheme_indices, dtype=tf.int32)
    # grapheme_embs = grapheme_emb(grapheme_indices)
    # print(grapheme_embs)


   except StopIteration:
    pass
