import gzip
import os
from pathlib import Path

import tensorflow as tf

def normalize(s: str):
    return s.lower().replace('_', ' ').strip()

class WikipediaTitlesDataset(tf.data.Dataset):
    def _generator(path: Path):
        print(f"WPTD initializing for {path}")
        with gzip.open(path, 'rt') as r:
            it = iter(r)

            _header = next(it)

            for line in it:
                normalized = normalize(line)
                print(f"WPTD normalized {normalized}")
                yield tf.constant(normalized, dtype=tf.string, shape=(None,))

    def __new__(cls, path: Path):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (None,), dtype = tf.string),
            args=(path,)
        )
