import sys
import os
import numpy as np

sys.path.append('./model')
sys.path.append('./data')
sys.path.append('./embed')

import data
import embed

#from rnn_model import RNNModel
from sep_cnn_model import SepCNNModel


TOP_K = 20000

def run_training():
  texts, labels = data.load_train_data()

  # create empedding
  input, word_index = embed.sequence_vectorize(texts)
  labels = np.array(labels)

  # create model
  model = SepCNNModel()

  # pipeline
  num_features = min(len(word_index) + 1, TOP_K)
  embedding_matrix = embed.get_embedding_matrix(word_index, embedding_dim=200)

  model.build(num_features, input.shape, 
              use_pretrained_embedding=True,
              is_embedding_trainable=False,
              embedding_matrix=embedding_matrix)

  model.fit(input, labels, epochs=10)
  model.save(f'saved_models/{type(model).__name__}')

# Predict
if __name__ == '__main__':
  run_training()
