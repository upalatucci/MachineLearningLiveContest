"""
   This script contain methods to read the TFRecord files. 
   extract_dataset allow to divide the 10 second video into different parts depending on the number of steps used. 
   Also the labels are repeated into the labels array. In this way we don't lost parts of dataset really importants
   when we want to use different step size to train differents models.  
"""

import tensorflow as tf
import os
import numpy as np

UNBAL_TRAIN_DIRECTORY = "./unbal_train/"
BAL_TRAIN_DIRECTORY = "./bal_train/"
EVAL_DIRECTORY = "./eval/"

def parse(serialized):
  context_features = {
      'labels': tf.VarLenFeature(tf.int64)
  }

  feature_list = {
      'audio_embedding': tf.FixedLenSequenceFeature([], tf.string)
  }

  # Parse the serialized data so we get a dict with our data.
  context_parsed, sequence_parsed = tf.parse_single_sequence_example(serialized=serialized,
                                    context_features=context_features, sequence_features=feature_list)
  sequence_parsed["audio_embedding"] = tf.decode_raw(sequence_parsed["audio_embedding"], tf.uint8)

  return context_parsed, sequence_parsed


def get_iterator_for_directory(path):
  dataset = tf.data.TFRecordDataset(filenames=[os.path.join(path, f) for f in os.listdir(path)])
  dataset = dataset.map(parse)
  iterator = dataset.make_one_shot_iterator()
  return iterator.get_next()


def extract_dataset(path, classes, steps):
  element = get_iterator_for_directory(path)
  n_classes = len(classes)
  labels_array = np.zeros(n_classes)
  labels = []
  audio_features = []
  weight_labels=[]
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
      while True:
        data_record = sess.run(element)
        labels_batch = data_record[0]["labels"].values
        audio_batch = data_record[1]["audio_embedding"]

        for label in labels_batch:
          if label in classes:
            weight_labels.append(label)
            labels_array[classes.index(label)] = 1

        if sum(labels_array) != 0:
            for i in range(audio_batch.shape[0]//steps):
                labels.append(labels_array)
                audio_features.append(audio_batch[i*steps:(i+1)*steps])

        labels_array = np.zeros(n_classes)
    except tf.errors.OutOfRangeError:
      pass
    except Exception as e:
      print(e)
    return np.array(audio_features), np.array(labels), weight_labels


class_to_consider = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]
classes_num = len(class_to_consider)



