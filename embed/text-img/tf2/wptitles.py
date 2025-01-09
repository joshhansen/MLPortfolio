import gzip
import os
from pathlib import Path

import tensorflow as tf
import tensorflow_text as tft

from grapheme_idx import GraphemeIdx

def normalize(s: str):
    return s.lower().replace('_', ' ').strip()

def _gen_wp_titles_dataset(path: Path, tokenizer: tft.WhitespaceTokenizer, grapheme_idx: GraphemeIdx, token_truncate_len: int):
    print(f"WPTD initializing for {path}")
    with gzip.open(path, 'rt') as r:
        it = iter(r)

        _header = next(it)

        for line in it:
            normalized = normalize(line)
            print(f"WPTD normalized {normalized}")

            trunc = normalized[:token_truncate_len]
            print(f"WPTD trunc {trunc}")

            grapheme_indices = grapheme_idx.index_tokens([trunc], pad_to=token_truncate_len)
            
            print(f"WPTD grapheme_indices: {grapheme_indices}")
            # tokens = tokenizer.tokenize(trunc)
            # print(f"WPTD tokens {tokens}")
            # print(f"WPTD token type: {type(tokens[0])}")
            # print(f"WPTD token numpy: {tokens.numpy()}")
            # tokens_as_indexed_graphemes = grapheme_idx.index_tokens(tokens.numpy(), pad_to=token_truncate_len)
            # print(f"WPTD tokens_as_indexed_graphemes: {tokens_as_indexed_graphemes}")
            # yield tokens_as_indexed_graphemes 
            yield grapheme_indices
            

            # yield normalized
            # yield tf.constant(normalized, dtype=tf.string)

def wp_titles_dataset(path: Path, tokenizer: tft.WhitespaceTokenizer, grapheme_idx: GraphemeIdx, token_truncate_len: int):
    gen = lambda: _gen_wp_titles_dataset(path, tokenizer, grapheme_idx, token_truncate_len)
    return tf.data.Dataset.from_generator(
        gen,
        output_signature = tf.TensorSpec(shape=(None,token_truncate_len+2,), dtype=tf.int32),
    )
    
class WikipediaTitlesDataset(tf.data.Dataset):
    def _generator(path: Path):
        print(f"WPTD initializing for {path}")
        with gzip.open(path, 'rt') as r:
            it = iter(r)

            _header = next(it)

            for line in it:
                normalized = normalize(line)
                print(f"WPTD normalized {normalized}")
                yield normalized
                # yield tf.constant(normalized, dtype=tf.string)

    def __new__(cls, path: Path):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (), dtype = tf.string),
            args=(path,)
        )

class WikipediaTitlesTokenizedDataset(tf.data.Dataset):
    def _generator(path: Path, tokenizer: tft.WhitespaceTokenizer):
        print(f"WPTID initializing for {path}")
        titles = WikipediaTitlesDataset(path)
        for title in titles:
            title_s = title.numpy()
            print(f"title_s: {title_s}")

            yield title

    def __new__(cls, path: Path, tokenizer: tft.WhitespaceTokenizer):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.RaggedTensorSpec(shape = (None,None,), dtype = tf.string),
            args=(path, tokenizer,)
        )

class WikipediaTitlesTokenizedIndexedDataset(tf.data.Dataset):
    def _generator(path: Path, grapheme_idx: GraphemeIdx):
        print(f"WPTID initializing for {path}")
        titles = WikipediaTitlesDataset(path)
        for title in titles:
            title_s = title.numpy()
            print(f"title_s: {title_s}")

            yield title

    def __new__(cls, path: Path, grapheme_idx: GraphemeIdx):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (None,), dtype = tf.string),
            args=(path, grapheme_idx,)
        )
