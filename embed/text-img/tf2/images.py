import gzip
import os
from pathlib import Path

import tensorflow as tf

from PIL import UnidentifiedImageError


def _gen_images_dataset(path: Path, category: str, rescale_width: int, rescale_height: int):
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
            try:
                img = tf.keras.utils.load_img(img_path)

                img = tf.image.resize(img, (rescale_width, rescale_height))

                yield img
            except UnidentifiedImageError as e:
                print(f"Error loading image: {img_path}: {e}")

def images_dataset(path: Path, category: str, rescale_width: int, rescale_height: int):
    gen = lambda: _gen_images_dataset(path, category, rescale_width, rescale_height)

    return tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(None, None, 3,), dtype=tf.float32))
