import tensorflow as tf
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight


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


def extract_dataset(path, classes, steps=2):
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


UNBAL_TRAIN_DIRECTORY = "../audioset_v1_embeddings/unbal_train/"
BAL_TRAIN_DIRECTORY = "../audioset_v1_embeddings/bal_train/"
EVAL_DIRECTORY = "../audioset_v1_embeddings/eval/"

class_to_consider = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]
classes_num = len(class_to_consider)

#audio_features_train, labels_train, class_weights_train = extract_dataset(UNBAL_TRAIN_DIRECTORY, class_to_consider)
#audio_features_bal_train, labels_bal_train, weight_bal_train = extract_dataset(BAL_TRAIN_DIRECTORY, class_to_consider)
#class_weights_train = class_weight.compute_class_weight('balanced', np.unique(class_weights_train),class_weights_train)
#weight_bal_train = class_weight.compute_class_weight('balanced', np.unique(weight_bal_train),weight_bal_train)


from src.custom_multiclassifier import CustomMulticlassifier

multiclass = CustomMulticlassifier("configuration.json")

for i in range(1, 11):
    audio_features_val, labels_val, class_weights_val = extract_dataset(EVAL_DIRECTORY, class_to_consider, i)
    labels_max = np.argmax(labels_val, axis=1)

    labels = []
    for l in labels_max:
      labels.append(class_to_consider[l])

    prediction = []

    for audio in audio_features_val:
      prediction.append(multiclass.predict(audio)[0])

    report = classification_report(labels, prediction)

    print("Report for {} steps".format(i))
    print(report)

    cm = confusion_matrix(labels, prediction)
    print(cm)
