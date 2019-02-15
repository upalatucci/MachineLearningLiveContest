"""
    This file contain the code used to train the final models.
    In the project we used different models so we have different hyperparameter that we change
    in order to build the target model.

    As you can see in the documentation of the project we use different time_steps to predict different audio array.
    So, if we change timesteps we need also to change the number of epochs. Infact with a very large number of
    time_steps we need also, to train the model properly, a major number of epochs.
"""
import keras.backend as K
import numpy as np
from keras.initializers import VarianceScaling
from keras.layers import (Input, Dense, BatchNormalization, Dropout, Lambda,
                          Activation, Concatenate)
from keras.models import Model
from keras.optimizers import Adam
from sklearn.utils import class_weight

from read_tfrecords import extract_dataset, UNBAL_TRAIN_DIRECTORY, BAL_TRAIN_DIRECTORY, EVAL_DIRECTORY

class_weights_unbal = [ 0.96078073,  2.31306902,  6.9915668,   4.85723096,  3.35935167,  2.47322971,
                    0.39771752,  1.40655416,  0.6240393,  0.33085486, 11.48942378,  0.64991542]

class_weights_bal = [1.1547619,  2.15555556, 2.15555556, 2.15555556, 2.15555556, 2.02083333,
                        0.47374847, 1.06010929, 0.45221445, 0.6953405,  2.08602151, 0.62479871]

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
batch_size = 256
learning_rate = 0.001
time_steps = 1
freq_bins = 128
epochs_unbal = 20
epochs_bal = 10

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


audio_features_train, labels_train, class_weights_train = extract_dataset(UNBAL_TRAIN_DIRECTORY, class_to_consider)
audio_features_bal_train, labels_bal_train, weight_bal_train = extract_dataset(BAL_TRAIN_DIRECTORY, class_to_consider)
audio_features_val, labels_val, _ = extract_dataset(EVAL_DIRECTORY, class_to_consider)
class_weights_train = class_weight.compute_class_weight('balanced', np.unique(class_weights_train),class_weights_train)
weight_bal_train = class_weight.compute_class_weight('balanced', np.unique(weight_bal_train),weight_bal_train)

model.fit(audio_features_train, labels_train, class_weight=class_weights_train, epochs=epochs_unbal, validation_data=(audio_features_val, labels_val), batch_size=batch_size)

model.fit(audio_features_bal_train, labels_bal_train, class_weight=weight_bal_train, epochs=epochs_bal, validation_data=(audio_features_val, labels_val), batch_size=batch_size)

model.save("model.h5")