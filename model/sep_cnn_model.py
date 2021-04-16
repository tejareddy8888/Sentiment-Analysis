from base_model import BaseModel
import os
import numpy as np
import datetime

import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.layers import SeparableConv1D
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.layers import GlobalAveragePooling1D

from custom_callback import CustomCallback

checkpoint_path = "checkpoints/sepcnn/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

TOP_K = 20000

class SepCNNModel(BaseModel):

  def build(
      self,
      num_features,
      input_shape,
      use_pretrained_embedding,
      is_embedding_trainable,
      embedding_matrix,
      blocks=2,
      dropout_rate=0.1,
      embedding_dim=200,
      filters=64,
      kernel_size=3,
      pool_size=3):

    print("Build shape", num_features, input_shape)

    # create model
    self.model = models.Sequential()

    if use_pretrained_embedding:
      self.model.add(Embedding(input_dim=num_features,
                          output_dim=embedding_dim,
                          input_length=input_shape[1],
                          weights=[embedding_matrix],
                          trainable=is_embedding_trainable))
    else:
      self.model.add(Embedding(input_dim=num_features,
                               output_dim=embedding_dim,
                               input_length=input_shape[1]))

    for _ in range(blocks-1):
      self.model.add(Dropout(rate=dropout_rate))
      self.model.add(SeparableConv1D(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same'))
      self.model.add(SeparableConv1D(filters=filters,
                                kernel_size=kernel_size,
                                activation='relu',
                                bias_initializer='random_uniform',
                                depthwise_initializer='random_uniform',
                                padding='same'))
      self.model.add(MaxPooling1D(pool_size=pool_size))

    self.model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    self.model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    self.model.add(GlobalAveragePooling1D())
    self.model.add(Dropout(rate=dropout_rate))
    self.model.add(Dense(1, activation='sigmoid'))

    # compile model with loss and optimizer
    optimizer = tf.keras.optimizers.Adam(
      tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3, 
        decay_steps=200000, 
        decay_rate=0.9))

    self.model.compile(
      loss='binary_crossentropy',
      optimizer=optimizer,
      metrics=['acc', 'mae']
    )


  def fit(self, input, labels, epochs=1000, batch_size=512):
    early_stopping_cb = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    self.model.fit(
      input,
      labels,
      epochs=epochs,
      batch_size=batch_size,
      validation_split=0.1,
      callbacks=[CustomCallback(), early_stopping_cb, tensorboard_callback],
      verbose=2, 
    )

  def predict(self, prediction_data):
    predictions = self.model.predict(prediction_data)
    return predictions


  def save(self, path):
    self.model.save(path) 
