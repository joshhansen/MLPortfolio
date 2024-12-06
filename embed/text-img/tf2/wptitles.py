import gzip
import os
from pathlib import Path

import tensorflow as tf

def normalize(s: str):
    return s.lower().replace('_', ' ').strip()

class WikipediaTitlesDataset(tf.data.Dataset):
    def _generator(path: Path):
        with gzip.open(path, 'rt') as r:
            it = iter(r)

            _header = next(it)

            for line in it:
                yield tf.constant([normalize(line)], dtype=tf.string, shape=(1,))

    def __new__(cls, path: Path):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.string),
            args=(path,)
        )

