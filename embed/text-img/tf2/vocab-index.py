from collections import Counter
import itertools
import os
import sys

# import tensorflow_text as tft

# from t2i import T2I

# from gutenberg import GutenbergTextDataset
from wptitles import WikipediaTitlesDataset


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

if __name__=="__main__":
  home_dir = os.path.expanduser('~')

  # tokenizer = tft.UnicodeScriptTokenizer()
  # text = GutenbergTextDataset(text_dir)
  # text = text.map(lambda x: tft.ngrams(tokenizer.tokenize(x), 5, reduction_type=tft.Reduction.STRING_JOIN, string_separator='\x00'))
  # text = text.shuffle(200)

  wptitles_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0.gz')
  text = WikipediaTitlesDataset(wptitles_path)
  # text = text.map(lambda x: tokenizer.tokenize(x))

  grapheme_counts = Counter()

  # t2i = T2I()
  

  for i, title in enumerate(text):
    title = title.numpy()[0].decode('utf-8')
    for c in title:
      grapheme_counts[c] += 1

    if i % 1000 == 0:
      eprint(i)
      # eprint(len(grapheme_counts))
      # eprint(grapheme_counts.total())

    # if i > 100:
    #   break

  counts = list(grapheme_counts.items())

  counts.sort(key = lambda x: x[1], reverse=True)

  for k,v in counts:
    print(f"{k}\t{v}")


  # it_txt = iter(text)

  # it_txt = [x.numpy()[0].decode('utf-8') for x in itertools.islice(it_txt, 100)]

  # t2i = T2I.build(it_txt, delimiter='')
  # print(t2i)
  
  # # n = 0
  # # while True:
  # #  try:
  # #   b_txt = next(it_txt).numpy()[0]
  # #   n += 1
    
  # #   txt = b_txt.decode('utf-8')
    
    
  # #   # print(txt)

  # #   for c in txt:
  # #    # print(c)
  # #    t2i.extend(c)

  # #   if n % 1000 == 0:
  # #    print(t2i)
     
  # #  except StopIteration:
  # #   pass

