import gzip
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_text as tft

from grapheme_idx import GraphemeIdx

def normalize(s: str) -> str:
    return s.lower().replace('_', ' ').strip()

def _gen_wp_titles_dataset(path: Path, tokenizer: tft.WhitespaceTokenizer, grapheme_idx: GraphemeIdx, token_truncate_len: int, include_label=False):
    print(f"WPTD initializing for {path}")
    with gzip.open(path, 'rt') as r:
        it = iter(r)

        _header = next(it)

        for line in it:
            normalized = normalize(line)
            # print(f"WPTD normalized {normalized}")

            if len(normalized) > token_truncate_len - 2:
             trunc = normalized[:(token_truncate_len - 1)]# -2 to allow start/end, +1 for exclusive range end
            else:
             trunc = normalized
                

            # print(f"WPTD trunc {trunc}")

            grapheme_indices = grapheme_idx.index_txt(trunc, max_output_len=token_truncate_len, use_unk=True)
            
            if include_label:
                yield grapheme_indices, grapheme_indices
            else:
                yield grapheme_indices


def wp_titles_dataset(path: Path, tokenizer: tft.WhitespaceTokenizer, grapheme_idx: GraphemeIdx, token_truncate_len: int, include_label=False):
    gen = lambda: _gen_wp_titles_dataset(path, tokenizer, grapheme_idx, token_truncate_len, include_label)

    if include_label:
        return tf.data.Dataset.from_generator(gen, output_signature=(
            tf.TensorSpec(shape=(token_truncate_len,), dtype=tf.int32),
            tf.TensorSpec(shape=(token_truncate_len,), dtype=tf.int32)
        ))

    return tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(token_truncate_len,), dtype=tf.int32))
