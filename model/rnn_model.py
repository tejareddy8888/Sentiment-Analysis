from base_model import BaseModel
import tensorflow as tf
import os
import numpy as np

from custom_callback import CustomCallback

checkpoint_path = "checkpoints/rnn/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

class RNNModel(BaseModel):

  def build(self, num_features):
    # create model
    self.model = tf.keras.Sequential([
      tf.keras.layers.Embedding(num_features, 64),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(1)
    ])

    # compile model with loss and optimizer
    self.model.compile(
      loss='binary_crossentropy',
      optimizer=tf.keras.optimizers.Adam(1e-4),
      metrics=['accuracy']
    )


  def fit(self, input, labels, epochs=3, batch_size=512):
    self.model.fit(
      input,
      labels,
      epochs=epochs,
      batch_size=batch_size,
      validation_split=0.1,
      callbacks=[CustomCallback()],
      verbose=2, 
    )


  def predict(self, prediction_data):
    predictions = self.model.predict(prediction_data)
    return predictions


  def save(self, path):
    self.model.save(path) 
