import tensorflow as tf
import os

import keras
from keras.models import Model, Sequential
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate, LSTM, TimeDistributed)
from keras.optimizers import Adam
from sklearn import metrics
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from sklearn import svm
  
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

def generator(path, batch_size, all_data=False):
  element = get_iterator_for_directory(path)
  n_classes = len(class_to_consider)
  labels_array = np.zeros(n_classes)
  labels = []
  audio_features = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
      while True:
        data_record = sess.run(element)
        labels_batch = data_record[0]["labels"].values
        audio_batch = data_record[1]["audio_embedding"]

        for label in labels_batch:
          if label in class_to_consider:
            labels_array[class_to_consider.index(label)] = 1

        if np.shape(audio_batch) == (10, 128):
          if sum(labels_array) != 0 or all_data:
            labels.append(labels_array)
            audio_features.append(audio_batch)

        if len(labels) >= batch_size:
          yield np.array(audio_features), np.array(labels)
          
        labels_array = np.zeros(n_classes)
    except tf.errors.OutOfRangeError:
      yield np.array(audio_features), np.array(labels)
    except Exception as e:
      print(e)

def extract_dataset(path, batch_size, all_data=False):
  element = get_iterator_for_directory(path)
  n_classes = len(class_to_consider)
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
          if label in class_to_consider:
            weight_labels.append(label)
            labels_array[class_to_consider.index(label)] = 1

        if np.shape(audio_batch) == (10, 128):
          if sum(labels_array) != 0 or all_data:
          	for i in range(audio_batch.shape[0]):
	            labels.append(labels_array)
	            audio_features.append(audio_batch[i])

        labels_array = np.zeros(n_classes)
    except tf.errors.OutOfRangeError:
      pass
    except Exception as e:
      print(e)
    return np.array(audio_features), np.array(labels), weight_labels
   

UNBAL_N_SAMPLES = 2013208
BAL_N_SAMPLES = 21782
EVAL_N_SAMPLES = 19976

UNBAL_TRAIN_DIRECTORY = "../audioset_v1_embeddings/unbal_train/"
BAL_TRAIN_DIRECTORY = "../audioset_v1_embeddings/bal_train/"
EVAL_DIRECTORY = "../audioset_v1_embeddings/eval/"

class_to_consider = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]
classes_num = len(class_to_consider)

  
def attention_pooling(inputs, **kwargs):
  [out, att] = inputs

  epsilon = 1e-7
  att = K.clip(att, epsilon, 1. - epsilon)
  normalized_att = att / K.sum(att, axis=1)[:, None, :]

  return K.sum(out * normalized_att, axis=1)
  
  
def pooling_shape(input_shape):
  if isinstance(input_shape, list):
      (sample_num, time_steps, freq_bins) = input_shape[0]

  else:
      (sample_num, time_steps, freq_bins) = input_shape

  return sample_num, freq_bins


# Hyper parameters
hidden_units = 1024
drop_rate = 0.5
batch_size = 100
learning_rate = 0.001
time_steps = 10
freq_bins = 128
n_total_classes = classes_num

# Embedded layers

input_layer = Input(shape=(time_steps, freq_bins))

a1 = Dense(hidden_units)(input_layer)
a1 = BatchNormalization()(a1)
a1 = Activation('relu')(a1)
a1 = Dropout(drop_rate)(a1)

a2 = Dense(hidden_units)(a1)
a2 = BatchNormalization()(a2)
a2 = Activation('relu')(a2)
a2 = Dropout(drop_rate)(a2)

a3 = Dense(hidden_units)(a2)
a3 = BatchNormalization()(a3)
a3 = Activation('relu')(a3)
a3 = Dropout(drop_rate)(a3)

cla1 = Dense(n_total_classes, activation='sigmoid')(a2)
att1 = Dense(n_total_classes, activation='softmax')(a2)
out1 = Lambda(
  attention_pooling, output_shape=pooling_shape)([cla1, att1])

cla2 = Dense(n_total_classes, activation='sigmoid')(a3)
att2 = Dense(n_total_classes, activation='softmax')(a3)
out2 = Lambda(
  attention_pooling, output_shape=pooling_shape)([cla2, att2])

b1 = Concatenate(axis=-1)([out1, out2])
b1 = Dense(n_total_classes)(b1)
output_layer = Activation('sigmoid')(b1)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = Adam(lr=learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("Model compiled")


#audio_features_train, labels_train, class_weights_train = extract_dataset(UNBAL_TRAIN_DIRECTORY, class_to_consider)
audio_features_val, labels_val, class_weights_val = extract_dataset(EVAL_DIRECTORY, class_to_consider)
audio_features_bal_train, labels_bal_train, weight_bal_train = extract_dataset(BAL_TRAIN_DIRECTORY, class_to_consider)
#class_weights_train = class_weight.compute_class_weight('balanced', np.unique(class_weights_train),class_weights_train)
weight_val = class_weight.compute_class_weight('balanced', np.unique(class_weights_val),class_weights_val)
weight_bal_train = class_weight.compute_class_weight('balanced', np.unique(weight_bal_train),weight_bal_train)
