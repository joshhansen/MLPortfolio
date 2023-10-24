from typing import Mapping

from collections import Counter
import itertools
import json
import os
import re
import time

from more_itertools import unzip

import numpy as np



np.random.seed(48349834)

np_rng = np.random.default_rng()
def random_split(data, weights):
 parts = [ list() for _ in weights ]

 for datum in data:
  x = np_rng.random()

  total = 0.0
  for i, weight in enumerate(weights):
   total += weight

   if x <= total:
    parts[i].append(datum)
    break

 return parts


stopwords = set([
 'i',
 'me',
 'my',
 'myself',
 'we',
 'our',
 'ours',
 'ourselves',
 'you',
 'your',
 'yours',
 'yourself',
 'yourselves',
 'he',
 'him',
 'his',
 'himself',
 'she',
 'her',
 'hers',
 'herself',
 'it',
 'its',
 'itself',
 'they',
 'them',
 'their',
 'theirs',
 'themselves',
 'what',
 'which',
 'who',
 'whom',
 'this',
 'that',
 'these',
 'those',
 'am',
 'is',
 'are',
 'was',
 'were',
 'be',
 'been',
 'being',
 'have',
 'has',
 'had',
 'having',
 'do',
 'does',
 'did',
 'doing',
 'a',
 'an',
 'the',
 'and',
 'but',
 'if',
 'or',
 'because',
 'as',
 'until',
 'while',
 'of',
 'at',
 'by',
 'for',
 'with',
 'about',
 'against',
 'between',
 'into',
 'through',
 'during',
 'before',
 'after',
 'above',
 'below',
 'to',
 'from',
 'up',
 'down',
 'in',
 'out',
 'on',
 'off',
 'over',
 'under',
 'again',
 'further',
 'then',
 'once',
 'here',
 'there',
 'when',
 'where',
 'why',
 'how',
 'all',
 'any',
 'both',
 'each',
 'few',
 'more',
 'most',
 'other',
 'some',
 'such',
 'no',
 'nor',
 'not',
 'only',
 'own',
 'same',
 'so',
 'than',
 'too',
 'very',
 's',
 't',
 'can',
 'will',
 'just',
 'don',
 'should',
 'now',
])

token_rgx = re.compile("[A-Za-z0-9]+")

def tokenize(s):
 return [s.lower() for s in token_rgx.findall(s) if s.lower() not in stopwords]

def load():
 path =  os.environ['PWD'] + "/../imdb-data/reviews.json"

 with open(path) as r:
  raw = json.load(r)
 
 vocab = set()
 vocab.add("__padding__")
 for datum in raw:
  words = tokenize(datum['t'])
  vocab.update(words)

 vocab = list(vocab)
 vocab.sort()

 vocab_len = len(vocab)
 print(f"vocab_len: {vocab_len}")

 word_to_idx = {word: i for i, word in enumerate(vocab)}
 padding_idx = word_to_idx["__padding__"]

 del vocab

 indexed: list[tuple[ list[int], int]] = list()
 lens: Counter = Counter()

 for datum in raw:
  words = tokenize(datum['t'])
  word_indices = [ word_to_idx[word] for word in words ]

  lens[len(word_indices)] += 1

  class_ = datum['s']
  indexed.append((word_indices, class_))
 
 del raw
 del word_to_idx

 sorted_lens: list[tuple[int,int]] = list(lens.items())
 sorted_lens.sort(key = lambda x: x[0])
 total_lens = lens.total()
 del lens

 cum = 0
 target_len = -1
 for l, n in sorted_lens:
  cum += n
  pct = cum / total_lens
  if pct >= 0.95:
   target_len = l
   break

 print(f"target_len: {target_len}")
 print(f"padding_idx: {padding_idx}")

 del sorted_lens

 data = list()
 for x, y in indexed:
  # Pad to target_len
  if len(x) < target_len:
   x.extend([padding_idx] * (target_len - len(x)))
  else:
   x = x[:target_len]

  data.append((x, y))

 del indexed
 
 print(f"data: {len(data)}")

 train, val, test = random_split(data, [0.8, 0.1, 0.1])
 del data

 print(f"train: {len(train)}")
 print(f"val: {len(val)}")
 print(f"test: {len(test)}")

 x_train, y_train = unzip(train)
 x_val, y_val = unzip(val)
 x_test, y_test = unzip(val)

 return {
  'x_train': list(x_train),
  'y_train': list(y_train),
  'x_val': list(x_val),
  'y_val': list(y_val),
  'x_test': list(x_test),
  'y_test': list(y_test),
  'vocab_len': vocab_len,
 }
