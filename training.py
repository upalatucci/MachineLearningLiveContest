"""
    This file contain the code used to train the final models.
    In the project we used different models so we have different hyperparameter that we change
    in order to build the target model.

    As you can see in the documentation of the project we use different time_steps to predict different audio array.
    
"""
import keras.backend as K
import numpy as np
from keras.initializers import VarianceScaling
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.utils import class_weight

from read_tfrecords import extract_dataset, UNBAL_TRAIN_DIRECTORY, BAL_TRAIN_DIRECTORY, EVAL_DIRECTORY


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

  return (sample_num, freq_bins)


# Hyper parameters

class_to_consider = [16, 23, 47, 49, 53, 67, 74, 81, 288, 343, 395, 396]
classes_num = len(class_to_consider)
hidden_units = 1024
drop_rate = 0.4
batch = 256
learning_rate = 0.001
time_steps = 3 
freq_bins = 128
epoch = 30

# Embedded layers
input_layer = Input(shape=(time_steps, freq_bins))

a1 = Dense(hidden_units,kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(input_layer)
a1 = BatchNormalization()(a1)
a1 = Activation('relu')(a1)
a1 = Dropout(drop_rate)(a1)

a2 = Dense(hidden_units,kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a1)
a2 = BatchNormalization()(a2)
a2 = Activation('relu')(a2)
a2 = Dropout(drop_rate)(a2)

a22 = Dense(hidden_units,kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a2)
a22 = BatchNormalization()(a22)
a22 = Activation('relu')(a22)
a22 = Dropout(drop_rate)(a22)

a3 = Dense(hidden_units,kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a22)
a3 = BatchNormalization()(a3)
a3 = Activation('relu')(a3)
a3 = Dropout(drop_rate)(a3)


cla = Dense(classes_num, activation='sigmoid',kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a2)
att = Dense(classes_num, activation='softmax',kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a2)
out = Lambda(
  attention_pooling, output_shape=pooling_shape)([cla, att])

cla1 = Dense(classes_num, activation='sigmoid',kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a22)
att1 = Dense(classes_num, activation='softmax',kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a22)
out1 = Lambda(
  attention_pooling, output_shape=pooling_shape)([cla1, att1])

cla2 = Dense(classes_num, activation='sigmoid',kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a3)
att2 = Dense(classes_num, activation='softmax',kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(a3)
out2 = Lambda(
  attention_pooling, output_shape=pooling_shape)([cla2, att2])


b1 = Concatenate(axis=-1)([out,out1, out2])
b1 = Dense(classes_num,kernel_initializer=VarianceScaling(scale=2.0, mode='fan_in', distribution='normal', seed=None))(b1)
output_layer = Activation('sigmoid')(b1)

model = Model(inputs=input_layer, outputs=output_layer)

optimizer = Adam(lr=learning_rate)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

print("Model compiled")

audio_features_train_unbal, labels_train_unbal, class_weights_unbal = extract_dataset(UNBAL_TRAIN_DIRECTORY, class_to_consider,time_steps)
audio_features_train_bal, labels_train_bal, class_weights_bal = extract_dataset(BAL_TRAIN_DIRECTORY, class_to_consider,time_steps)
audio_features_val, labels_val, _ = extract_dataset(EVAL_DIRECTORY, class_to_consider,time_steps)

# Merge of datasets
 

labels_train_unbal_bal = []
audio_features_train_unbal_bal = []

for i in audio_features_train_unbal:
  audio_features_train_unbal_bal.append(i)
for i in labels_train_unbal:
  labels_train_unbal_bal.append(i)
  
for i in audio_features_train_bal:
  audio_features_train_unbal_bal.append(i)
for i in labels_train_bal:
  labels_train_unbal_bal.append(i) 

class_weight_unbal_bal=[]
for i in class_weights_unbal:
  class_weight_unbal_bal.append(i)
for i in class_weights_bal:
  class_weight_unbal_bal.append(i)

class_weights = class_weight.compute_class_weight('balanced', np.unique(class_weight_unbal_bal),class_weight_unbal_bal)

early_stop = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=5, verbose=0, mode='auto')

history = model.fit(np.array(audio_features_train_unbal_bal),np.array(labels_train_unbal_bal), validation_data=(audio_features_val, labels_val), callbacks = [early_stop], epochs=epoch, batch_size=batch, class_weight = class_weights, verbose=0)

model.save("model.h5")