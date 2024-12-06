import os

import tensorflow as tf
import tensorflow_text as tft

from t2i import T2I

from gutenberg import GutenbergTextDataset
from wptitles import WikipediaTitlesDataset

if __name__=="__main__":
 with tf.device('/CPU:0'):
  home_dir = os.path.expanduser('~')
  text_dir = os.path.join(home_dir, 'Data', 'org', 'gutenberg', 'mirror_txt')
  img_dir = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'wikimedia-commons-hires-png_not-too-big')

  tokenizer = tft.UnicodeScriptTokenizer()
  # text = GutenbergTextDataset(text_dir)
  # text = text.map(lambda x: tft.ngrams(tokenizer.tokenize(x), 5, reduction_type=tft.Reduction.STRING_JOIN, string_separator='\x00'))
  # text = text.shuffle(200)

  wptitles_path = os.path.join(home_dir, 'Data', 'org', 'wikimedia', 'enwiki-20241201-all-titles-in-ns0.gz')
  text = WikipediaTitlesDataset(wptitles_path)
  text = text.map(lambda x: tokenizer.tokenize(x))

  images = tf.keras.preprocessing.image_dataset_from_directory(
   img_dir,
   labels=None,
   label_mode=None,
   crop_to_aspect_ratio=True,
   shuffle=False,
   batch_size=10
  )
  # images = images.shuffle(200)

  # data = tf.data.Dataset.sample_from_datasets([text, images], [0.5, 0.5], seed=8439389)
  # data = data.shuffle(200)

  # for datum in data:
  #  print(datum)
  # pass
  EMBED = 64
  WORD_EMBED = EMBED

  word_emb = tf.keras.layers.Embedding(
  )
  txt2emb = tf.keras.models.Sequential([
   tf.keras.layers.Flatten(input_shape=(28, 28)),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(10, activation='softmax')
  ])
  emb2txt = tf.keras.models.Sequential([
   tf.keras.layers.Flatten(input_shape=(28, 28)),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(10, activation='softmax')
  ])
  img2emb = tf.keras.models.Sequential([
   tf.keras.layers.Flatten(input_shape=(28, 28)),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(10, activation='softmax')
  ])
  emb2img = tf.keras.models.Sequential([
   tf.keras.layers.Flatten(input_shape=(28, 28)),
   tf.keras.layers.Dense(128, activation='relu'),
   tf.keras.layers.Dropout(0.2),
   tf.keras.layers.Dense(10, activation='softmax')
  ])

 model.compile(optimizer='adam',
   loss='sparse_categorical_crossentropy',
   metrics=['accuracy'])

 model.fit(x_train, y_train, epochs=5)
 model.evaluate(x_test, y_test)

  it_txt = iter(text)
  it_img = iter(images)
  while True:
   try:
    txt = next(it_txt)
    print(txt)
   except StopIteration:
    pass

   # try:
   #  img = next(it_img)
   #  print(img)
   # except StopIteration:
   #  pass


