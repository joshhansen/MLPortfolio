import os

import tensorflow as tf


BATCH_SIZE=10
SHUFFLE_SIZE=5000
AUDIO_LEN=64000
INSTR_FAM_EMBED_DIM=31


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
  del ex['audio']

  return ex, audio

 def dataset(ds):
  return ds.shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
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
 inputs = tf.keras.layers.Concatenate(axis=1)([instrument, pitch])
 repeated_inputs = tf.keras.layers.RepeatVector(AUDIO_LEN)(inputs)
 y = repeated_inputs

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

 model = tf.keras.Model([instrument, pitch], y)
 print(model.summary())

 #TODO mu-law

 model.compile(
  optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
  loss = tf.keras.losses.MeanSquaredError(),
  metrics = [tf.keras.metrics.MeanAbsolutePercentageError()]
 )
 
 history = model.fit(
  train,
  batch_size=BATCH_SIZE,
  epochs=50,
  validation_data = val,
  validation_steps = 5000,
  validation_freq = 2
 )
