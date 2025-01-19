import gzip
import os
from pathlib import Path

import tensorflow as tf


def _gen_images_dataset(path: Path, category: str):
    print(f"Images initializing for {path}")

    for i, f in enumerate(os.listdir(path)):
        img_path = os.path.join(path, f)

        result = False
        
        partition = i % 16
        if category == 'test':
            if partition == 0:
                result = True
        elif category == 'valid':
            if partition == 1:
                result = True
        elif category == 'train':
            if partition > 1:
                result = True

        if result:
            yield tf.keras.utils.load_img(img_path)

def images_dataset(path: Path, category: str):
    gen = lambda: _gen_images_dataset(path, category)

    return tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(None, None, 3,), dtype=tf.int8))
