import os
from pathlib import Path

import tensorflow as tf

def normalize(s: str):
    return s.lower()

class GutenbergTextDataset(tf.data.Dataset):
    def _generator(path: Path):
        for root, dirs, files in os.walk(path):
            # path = root.split(os.sep)
            # print((len(path) - 1) * '---', os.path.basename(root))
            for file in files:
                # print(len(path) * '---', file)

                path = os.path.join(root, file)

                if path.endswith(b'-8.txt'):
                    continue

                if b'/old' in path:
                    continue

                if path.endswith(b'-0.txt'):
                    continue

                print(path)

                try:
                    with open(path, 'r') as r:
                        yield tf.constant([normalize(r.read())], dtype=tf.string, shape=(1,))
                except FileNotFoundError:
                    print(f"File not found: {path}")
                except UnicodeDecodeError:
                    with open(path, 'r', encoding='latin1') as r:
                        yield tf.constant([normalize(r.read())], dtype=tf.string, shape=(1,))
    

    def __new__(cls, path: Path):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature = tf.TensorSpec(shape = (1,), dtype = tf.string),
            args=(path,)
        )

