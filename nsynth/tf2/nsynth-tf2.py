import os

import tensorflow as tf


BATCH_SIZE=100
SHUFFLE_SIZE=10000

def parse_example(raw):
 return tf.io.parse_example(raw, {
  'audio': tf.io.RaggedFeature(dtype=tf.float32),
  'pitch': tf.io.FixedLenFeature((), tf.int64),
  'sample_rate': tf.io.FixedLenFeature((), tf.int64),
 })

def dataset(ds):
 return ds.map(parse_example).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).take(10)

if __name__ == "__main__":
 data_dir = os.path.join(os.environ['HOME'], 'Data/org/tensorflow/magenta')
 train_path = os.path.join(data_dir, 'nsynth-train.tfrecord')
 val_path = os.path.join(data_dir, 'nsynth-valid.tfrecord')
 test_path = os.path.join(data_dir, 'nsynth-test.tfrecord')

 print(f"train_path: {train_path}")

 train = dataset(tf.data.TFRecordDataset(train_path))
 val = dataset(tf.data.TFRecordDataset(val_path))
 test = dataset(tf.data.TFRecordDataset(test_path))

 def parse_example(raw):
  return tf.io.parse_example(raw, {
   'audio': tf.io.RaggedFeature(dtype=tf.float32),
   'pitch': tf.io.FixedLenFeature((), tf.int64),
   'sample_rate': tf.io.FixedLenFeature((), tf.int64),
  })

 x = tf.keras.Input((64000,1,), BATCH_SIZE)
 y = x

 for block in range(3):
  for layer in range(8):
   dilation = 2**layer

   y = tf.keras.layers.Conv1D(
    32,
    2,
    dilation_rate=dilation,
    padding='causal',
    name=f"block{block}layer{layer}dilation{dilation}"
   )(y)

 model = tf.keras.Model(x, y)

 #TODO mu-law

 model.compile(
  optimizer=tf.keras.optimizers.RMSprop(),  # Optimizer
  # Loss function to minimize
  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
  # List of metrics to monitor
  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
 )

 # for batch in train.map(parse_example).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).take(10):
 #  print(batch['audio'].shape)
 #  print(batch['pitch'])
 #  print(batch['sample_rate'])

  
 history = model.fit(
    { 'input_1': train },
    batch_size=BATCH_SIZE,
    epochs=2,
    # validation_data=(val, val),
 )
