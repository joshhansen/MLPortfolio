import os

class GraphemeIdx:
 def __init__(self, idx: dict[str,int], start: str, end: str, pad: str, unk: str):
  self.idx = idx
  self.rev = dict()
  for k,v in idx.items():
   self.rev[v] = k

  self.index_grapheme(start)
  self.index_grapheme(end)
  self.index_grapheme(pad)
  self.unk_idx = self.index_grapheme(unk)

  self.start = start
  self.end = end
  self.pad = pad
  self.unk = unk
  self.special_grapheme_count = 4

 # use_unk: if the grapheme hasn't been indexed, should the unk index be returned? Otherwise, the grapheme will be indexed and its newly-assigned index will be returned
 def index_grapheme(self, grapheme: str, use_unk=False) -> int:
  try:
   return self.idx[grapheme]
  except KeyError:
   if use_unk:
    return self.unk_idx

   i = len(self.idx)
   self.idx[grapheme] = i
   self.rev[i] = grapheme
   return i

 def unindex_grapheme(self, grapheme_idx: int) -> str:
  return self.rev[grapheme_idx]
 
 def index_txt(self, txt: str, max_output_len: int, use_unk=False) -> list[int]:
  if len(txt) > max_output_len - 2:# -2 to allow for start and end tokens
   raise Exception(f"Tried to index text {txt} with max output len {max_output_len}; start/end might not fit")

  pads = max_output_len - len(txt) - 2# -2 to allow start/end tokens

  graphemes = [self.start] + list(txt) + [self.pad]*pads + [self.end]
  
  return [self.index_grapheme(g, use_unk=use_unk) for g in graphemes]

 def unindex_txt(self, token_grapheme_indices: list[int]) -> str:
  graphemes: list[str] = [self.unindex_grapheme(gi) for gi in token_grapheme_indices]
  return ''.join(graphemes)
   
 # Add all graphemes to the index, and return their indices
 def index_txts(self, txts: list[str], pad_to = 'max', use_unk=False) -> list[list[int]]:
  pad_to = max([len(t) for t in txts]) + 2 if pad_to=='max' else pad_to# +2 for start/end tokens
  return [self.index_txt(t, pad_to, use_unk=use_unk) for t in txts]

 def unindex_txts(self, token_grapheme_indices: list[list[int]]) -> list[str]:
  return [self.unindex_txt(tgi) for tgi in token_grapheme_indices]

 def __call__(self, txts: list[str]) -> list[list[int]]:
  return self.index_txts(txts)

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

  return GraphemeIdx(grapheme_idx, '<start>', '<end>', '<pad>', '<unk>')
