import os

import tensorflow as tf



if __name__ == "__main__":
 data_dir = os.path.join(os.environ['HOME'], 'Data/org/tensorflow/magenta')
 train_path = os.path.join(data_dir, 'nsynth-train.tfrecord')
 val_path = os.path.join(data_dir, 'nsynth-valid.tfrecord')
 test_path = os.path.join(data_dir, 'nsynth-test.tfrecord')

 print(f"train_path: {train_path}")

 train = tf.data.TFRecordDataset(train_path)
 for raw_record in train.take(1):
     example = tf.train.Example()
     example.ParseFromString(raw_record.numpy())
     print(example)

 
