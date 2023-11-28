import os

import tensorflow as tf


BATCH_SIZE=100
SHUFFLE_SIZE=10000

if __name__ == "__main__":
 data_dir = os.path.join(os.environ['HOME'], 'Data/org/tensorflow/magenta')
 train_path = os.path.join(data_dir, 'nsynth-train.tfrecord')
 val_path = os.path.join(data_dir, 'nsynth-valid.tfrecord')
 test_path = os.path.join(data_dir, 'nsynth-test.tfrecord')

 print(f"train_path: {train_path}")

 def parse_example(raw):
  return tf.io.parse_example(raw, {
   'audio': tf.io.RaggedFeature(dtype=tf.float32),
   'pitch': tf.io.FixedLenFeature((), tf.int64),
   'sample_rate': tf.io.FixedLenFeature((), tf.int64),
  })

 train = tf.data.TFRecordDataset(train_path)

 print(train.cardinality())

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

 for batch in train.map(parse_example).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE).take(10):
  print(batch['audio'].shape)
  print(batch['pitch'])
  print(batch['sample_rate'])

  

  

 
