#!/usr/bin/python3
from collections import Counter
import fileinput

MIN_PROPORTION = 0.9995

if __name__=="__main__":
 counts = Counter()

 for l in fileinput.input():
  grapheme, count = l.split('\t')
  count = count[:-1]
  count = int(count)

  counts[grapheme] = count

 total = counts.total()

 counts = list(counts.items())

 counts.sort(key=lambda x: x[1], reverse=True)

 proportion = 0.0

 for g, c in counts:
  rel_c = c / total

  proportion += rel_c

  print(f"{g}\t{c}\t{rel_c}\t{proportion}")

  if proportion >= MIN_PROPORTION:
   break

