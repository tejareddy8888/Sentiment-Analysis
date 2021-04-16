import sys
import os
import numpy as np

sys.path.append('./model')
sys.path.append('./data')
sys.path.append('./embed')
sys.path.append('./metrics')
import data
from san_model_pytorch import SentimentSAN
from sklearn.model_selection import train_test_split
from torch.cuda import is_available
from torch import device

## comment this
import argparse

parser = argparse.ArgumentParser(description='Sentiment Analysis Parameters')
parser.add_argument('-val_split', type=float, default=0.1,help='validation split')
parser.add_argument('-batch_size', type=int, default=64,help='Batch size val')
parser.add_argument('-epochs', type=int, default=1,help='Number of epochs')
parser.add_argument('-lr', type=float, default=0.0001,help='Learning rate')
parser.add_argument('-lr_decay_rate', type=float, default=0.9,help='LR decay rate')
args = parser.parse_args()



def run_training():

  train_df , test_df = data.fetch_and_load_datasets('pipeline_mmst')

  train_df, val_df = train_test_split(train_df, shuffle=True, train_size=1-args.val_split, test_size=args.val_split, random_state=42)

  train_df.reset_index(drop=True,inplace=True)

  val_df.reset_index(drop=True,inplace=True)

  Device = device("cuda" if is_available() else "cpu")

  hidden_dim = 300

  embedding_dim = 768

  output_dim = 2

  dropout_prob = 0.5

  num_of_layer = 2

  # create model
  model = SentimentSAN.build(train_df, val_df, 'GloVe', embedding_dim, hidden_dim, output_dim, args.batch_size, num_of_layer, dropout_prob, Device)

  model.fit(args.epochs,args.lr,args.lr_decay_rate)

  model.predict(test_df)

  model.save('saved_model/model_whole.pt')

# Predict
if __name__ == '__main__':
  run_training()
