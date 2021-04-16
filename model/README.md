## Structure

In this folder we keep multiple different models that we use for prototyping and classification. All models inherit from a base model that provides the interface for training, evaluation and prediction.

```
  .
  ├── checkpoints/                  # artifactories from training
  ├── base_model.py                 # base model that provides interface for all models
  ├── bigru_model_pytorch.py        # model with BiGRU layers and Glove/BERT embeddings
  ├── bilstm_model_pytorch.py       # model with BiLSTM layers and Glove/BERT embeddings
  ├── cil_pretrainingwithBERT       # prototyping model with BERT and dense layers
  ├── custom_callback.py            # callback that intercepts epochs and logs status
  ├── rnn_model.py                  # model with BILSTM and glove in rnn setting
  ├── san_model_pytorch.py          # model with BILSTM and self attention Glove/BERT
  ├── saved_model.py                # wrapper model for loading saved models
  ├── sep_cnn_model.py              # model with Separable convolutions and glove
  └── README.md
```

## Framework

For the models we used mainly [TensorFlow](https://www.tensorflow.org/api_docs/python/tf?version=nightly) and [PyTorch](https://pytorch.org/docs/stable/index.html). For the embedding we used [Glove](https://nlp.stanford.edu/projects/glove/) and [BERT](https://github.com/google-research/bert) as language models.

## Interface

All models implement the same base class and therefore can be used in a consistent way in the run scripts. The following base class with required functions is provided:

```
class BaseModel(ABC):

  @abstractmethod
  def build(self):
    pass

  @abstractmethod
  def fit(self, input):
    pass

  @abstractmethod
  def predict(self, input):
    pass
```

 - `build()`: Build the model depending on provided parameters
 - `fit()`: Train the model depending on runtime parameters
 - `predict()`: Predict labels on input

## Models

The following models were implemented:

 - [SepCNN](https://arxiv.org/abs/1610.02357) 
 - [BiLSTM with self attention](https://www.aclweb.org/anthology/P16-2034.pdf)
 - [GRU with self attention](https://arxiv.org/pdf/2002.00735.pdf)
 - [BiLSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)