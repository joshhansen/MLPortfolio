import os

class GraphemeIdx:
 def __init__(self, idx: dict[str,int], start: str, end: str, pad: str):
  self.idx = idx
  self.rev = dict()
  for k,v in idx.items():
   self.rev[v] = k

  self.index_grapheme(start)
  self.index_grapheme(end)
  self.index_grapheme(pad)

  self.start = start
  self.end = end
  self.pad = pad
  self.special_grapheme_count = 3

 def index_grapheme(self, grapheme: str) -> int:
  try:
   return self.idx[grapheme]
  except KeyError:
   i = len(self.idx)
   self.idx[grapheme] = i
   self.rev[i] = grapheme
   return i

 def unindex_grapheme(self, grapheme_idx: int) -> str:
  return self.rev[grapheme_idx]
 
 def index_token(self, token: str, pad_to_token_len: int) -> list[int]:
  if len(token) > pad_to_token_len:
   raise Error(f"Tried to index token longer than padding length")
  
  graphemes = [self.start] + list(token) + [self.end]
  pad_len = pad_to_token_len + 2 - len(graphemes)# +2 for start and end
  graphemes = graphemes + [self.pad] * pad_len

  return [self.index_grapheme(g) for g in graphemes]

 def unindex_token(self, token_grapheme_indices: list[int]) -> str:
  graphemes: list[str] = [self.unindex_grapheme(gi) for gi in token_grapheme_indices]
  return ''.join(graphemes)
   
 # Add all graphemes to the index, and return their indices
 def index_tokens(self, tokens: list[str], pad_to = 'max') -> list[list[int]]:
  pad_to = max([len(t) for t in tokens]) if pad_to=='max' else pad_to
  return [self.index_token(t, pad_to) for t in tokens]

 def unindex_tokens(self, token_grapheme_indices: list[list[int]]) -> list[str]:
  return [self.unindex_token(tgi) for tgi in token_grapheme_indices]

 def __call__(self, tokens: list[str]) -> list[list[int]]:
  self.index_tokens(tokens)

 def __str__(self) -> str:
  return str(self.idx)

 def __len__(self) -> int:
  return len(self.idx)

 def nonspecial_grapheme_count(self) -> int:
  return len(self) - self.special_grapheme_count

 def start_idx(self) -> int:
  return self.idx[self.start]
 def end_idx(self) -> int:
  return self.idx[self.end]
 def pad_idx(self) -> int:
  return self.idx[self.pad]
  
def load_grapheme_idx() -> GraphemeIdx:
  home_dir = os.path.expanduser('~')
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

  return GraphemeIdx(grapheme_idx, '<start>', '<end>', '<pad>')
