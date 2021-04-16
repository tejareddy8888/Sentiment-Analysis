# Sentiment analysis for Twitter
Short intro. We provide various preprocessing techniques as well as 4 models.

Please refer to XY paper for a description of the methods used, refer to Structure for a high level overview over the repository, refer to Prerequisits for installation instructions.

## Installation

For smooth running of the preprocessing and models we use pipenv library to manage virtual environments and dependencies on different devices. You can install via

1. Install `pipenv` with
```
pip install pipenv
```
2. Spin up a virtual environment with:
```
pipenv shell
```
3. Install dependencies (`--dev` option needed for some of the preprocessing steps)
```
pipenv install
```

## Structure

We divided the project into multiple folders corresponding to steps in the whole ml pipeline:

```
  .
  ├── data/                   # raw/processed datasets and data loaders
  ├── embed/                  # vocabularies and embeddings/vectorizers
  ├── model/                  # models and interceptors for training
  ├── preprocessing/          # pipelines and helpers for preprocessing data
  ├── run_predict.py          # script for loading saved models and predicting on test data
  ├── run_preprocessing.py    # script for executing preprocessing on data
  ├── run_training.py         # script for training model with TensorFlow
  ├── run_training_pytorch.py # script for training model with PyTorch
  └── README.md
```

### Script `run_preprocessing.py`

The preprocessing script is a playground where you can build up a complete preprocessing pipeline with the helper functions from the `/preprocessing` directory. All transformers follow the same interface, which simplifies chaining preprocessing steps.

See [Preproccessing](../preprocessing/README.md) for more information

### Script `run_training.py`

The training script loads the training data and a specific model from the `/model` directory. All models inherit the same base class to provide a consistent experience for this script. The models need to implement `build` and `fit` functions that take similar base parameters.

See [Model](../model/README.md) for more information

### Script `run_training_pytorch.py`

The training script loads the training data and a specific model from the `/model` directory with the PyTorch framework. All models inherit the same base class to provide a consistent experience for this script.

See [Model](../model/README.md) for more information

### Script `run_predict.py`

This script loads the saved models from the training and predicts classification responses from test data.

## Datasets
For convinience, we provide links to the already processed data sets as downloadable zip folders.
etc.

## Authors

This project was a collaboration for the Computational Intelligence Lab FS2020 at ETH Zurich from the following members:

Ferdinand Kossmann, Philippe Mösch, Saiteja Reddy Pottanigari, Kaan Sentürk
