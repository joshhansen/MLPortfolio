import numpy as np
import os

import tensorflow as tf


BATCH_SIZE=20
SHUFFLE_SIZE=5000
AUDIO_LEN=64000
INSTR_FAM_EMBED_DIM=30


if __name__ == "__main__":
 instrument_family_embedding = tf.keras.layers.Embedding(11, INSTR_FAM_EMBED_DIM, input_length=1)
 def parse_example(raw):
  ex = tf.io.parse_example(raw, {
   # 'input_1': tf.io.RaggedFeature(dtype=tf.float32),
   # 'input_1': {
    # 'audio': tf.io.RaggedFeature(dtype=tf.float32),
   'audio': tf.io.FixedLenFeature((AUDIO_LEN,), tf.float32),
   'instrument_family': tf.io.FixedLenFeature((), tf.int64),
   'pitch': tf.io.FixedLenFeature((), tf.int64),
   #  'sample_rate': tf.io.FixedLenFeature((), tf.int64),
   #  }
  })

  ex['instrument_family'] = instrument_family_embedding(ex['instrument_family'])

  audio = ex['audio']
  # del ex['audio']

  return ex, audio

 def subsample(i,x):
  return i % 20 == 0

 def dataset(ds):
  return ds.shuffle(SHUFFLE_SIZE).repeat().enumerate().filter(subsample).map(lambda i,x: x).batch(BATCH_SIZE)

 data_dir = os.path.join(os.environ['HOME'], 'Data/org/tensorflow/magenta')
 train_path = os.path.join(data_dir, 'nsynth-train.tfrecord')
 val_path = os.path.join(data_dir, 'nsynth-valid.tfrecord')
 test_path = os.path.join(data_dir, 'nsynth-test.tfrecord')

 print(f"train_path: {train_path}")

 train_raw = tf.data.TFRecordDataset(train_path).map(parse_example)
 val_raw = tf.data.TFRecordDataset(val_path).map(parse_example)
 test_raw = tf.data.TFRecordDataset(test_path).map(parse_example)

 train = dataset(train_raw)
 val = dataset(val_raw)
 test = dataset(test_raw)

 instrument = tf.keras.Input((INSTR_FAM_EMBED_DIM,), BATCH_SIZE, name="instrument_family")
 pitch = tf.keras.Input((1,), BATCH_SIZE, name="pitch")
 meta = tf.keras.layers.Concatenate(axis=1)([instrument, pitch])
 repeated_meta = tf.keras.layers.RepeatVector(AUDIO_LEN, name='repeated_meta')(meta)
 print(f"repeated meta shape: {repeated_meta.shape}")
 audio = tf.keras.Input((AUDIO_LEN, 1,), BATCH_SIZE, name="audio")
 print(f"audio shape: {audio.shape}")

 zero = tf.keras.backend.constant(np.zeros((BATCH_SIZE, 1, 1)))
 # audio_without_first = tf.keras.layers.Lambda(lambda x: x[:-1].concat(zero))
 # audio_without_first = audio[:-1].concat(zero)

 # Shift audio so convolutions don't have access to current audio output, only previous
 audio_without_first = tf.keras.layers.Concatenate(axis=1)([zero, audio[:, 1:]])

 audio_with_meta = tf.keras.layers.Concatenate(axis=2)([audio_without_first, repeated_meta])
 print(f"audio_with_meta shape: {audio_with_meta.shape}")

 y = audio_with_meta

 BLOCKS=3
 LAYERS=8

 for block in range(BLOCKS):
  for layer in range(LAYERS):
   dilation = 2**layer

   if block < BLOCKS-1:
    filters = 32
   else:
    filters = 2**(LAYERS - layer - 1)

   y = tf.keras.layers.Conv1D(
    filters,
    2,
    dilation_rate=dilation,
    padding='causal',
    name=f"block{block}layer{layer}dilation{dilation}"
   )(y)

 model = tf.keras.Model([audio, instrument, pitch], y)
 print(model.summary())

 #TODO mu-law

 output_dir = '/tmp/keras/checkpoint.model.tf'

 callbacks = [
  tf.keras.callbacks.ModelCheckpoint(
     filepath=output_dir,
     save_best_only=True
  )
 ]

 model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss = tf.keras.losses.MeanSquaredError(),
  metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]
 )
 
 history = model.fit(
  train,
  batch_size=BATCH_SIZE,
  epochs=50,
  steps_per_epoch=15000,
  validation_data = val,
  validation_steps = 1500,
  validation_freq = 1,
  callbacks=callbacks,
 )
