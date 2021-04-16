import sys
import os

sys.path.append('./model')
sys.path.append('./data')
sys.path.append('./preprocessing')

import data
import embed
from saved_model import SavedModel

MODEL_PATH = 'saved_models/rnn_model'

def run_predict():
  texts = data.load_test_data()
  input, word_index = embed.sequence_vectorize(texts)

  # Create a new model instance
  model = SavedModel()


  model.build(MODEL_PATH)

  # Re-evaluate the model
  predictions = model.predict(input)

# Predict
if __name__ == '__main__':
  run_predict()