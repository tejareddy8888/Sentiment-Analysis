from base_model import BaseModel
import tensorflow as tf
import os

checkpoint_path = "checkpoints/rnn/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

TOP_K = 20000

class SavedModel(BaseModel):

  def build(self, model_path):
    self.model = tf.keras.models.load_model(model_path)

  def fit(self, input, labels):
    print("LoadModel cannot be trained, since the model was already trained")

  def predict(self, input):
    predictions = self.model.predict(input)
    return predictions

  def save(self, path):
    self.model.save(path) 
