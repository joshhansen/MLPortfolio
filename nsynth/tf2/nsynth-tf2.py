import os

import tensorflow as tf


BATCH_SIZE=10
SHUFFLE_SIZE=100

def parse_example(raw):
 return tf.io.parse_example(raw, {
  # 'input_1': tf.io.RaggedFeature(dtype=tf.float32),
  # 'input_1': {
   # 'audio': tf.io.RaggedFeature(dtype=tf.float32),
  'audio': tf.io.FixedLenFeature((64000,), tf.float32)
  #  'pitch': tf.io.FixedLenFeature((), tf.int64),
  #  'sample_rate': tf.io.FixedLenFeature((), tf.int64),
  #  }
 })

def dataset(ds):
 return ds.map(lambda x: (x['audio'], x['audio'])).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

if __name__ == "__main__":
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

 x = tf.keras.Input((64000,1,), BATCH_SIZE)
 y = x

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

 model = tf.keras.Model(x, y)
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
 )
