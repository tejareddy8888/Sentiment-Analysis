import os
import sys
import random
from tqdm import tqdm
import pandas as pd

file_path = os.path.dirname(os.path.abspath(__file__))
get_prep_path = lambda file: os.path.join(file_path, f"./pipeline_mmst/{file}")

def load_train_data(seed = 0):
  # Gather data
  pos_path = get_prep_path("train_pos.txt")
  neg_path = get_prep_path("train_neg.txt")

  # Load the preprocessed training data
  features = []
  labels = []

  with open(pos_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)
      labels.append(1)

  with open(neg_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)
      labels.append(0)

  random.seed(seed)
  random.shuffle(features)
  random.shuffle(labels)

  return features, labels


def load_test_data():
  test_path = get_prep_path("test.txt")

  # Load the preprocessed test data
  features = []

  with open(test_path) as f:
    for line in f:
      line = line.strip()
      features.append(line)

  return features


def load_directory_data(file):
  """Retrieves the Sentences from the input text file into a Dict,stores and return into Dataframe """
  data = {}
  data["sentence"] = []
  num_lines = sum(1 for line in open(file,'r',encoding='utf-8'))
  with open(file,'r',encoding='utf-8') as f:
    print('\n opened file : '+file)
    for line in tqdm(f, total=num_lines):
        data["sentence"].append(line)
  return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory,Train):
  """ Specifically for assigning labels and get the train_df or test_df from the pre-defined datasets"""
  if (Train):
    pos_df = load_directory_data(os.path.join(directory, "train_pos_part.txt"))
    neg_df = load_directory_data(os.path.join(directory, "train_neg_part.txt"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
  else :
    test_df = load_directory_data(os.path.join(directory, "test.txt"))
    test_df["polarity"] = -1
  return test_df


def load_Pretraindataset():
  """ Specifically for assigning labels and get the train_df or test_df from the pre-defined datasets"""
  pos_df = load_directory_data("data/pretrain_data/pretrain_pos.txt")
  neg_df = load_directory_data("data/pretrain_data/pretrain_neg.txt")
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


# Fetch and Process the dataset files.
def fetch_and_load_datasets(folder):
  """ Initialises both train and test dataframes"""
  oldpwd=os.getcwd()
  os.chdir("data/"+folder+"/")
  train_df = load_dataset(os.getcwd(),True)
  test_df = load_dataset(os.getcwd(),False)
  os.chdir(oldpwd)
  return train_df.reset_index(drop=True) ,test_df.reset_index(drop=True)
