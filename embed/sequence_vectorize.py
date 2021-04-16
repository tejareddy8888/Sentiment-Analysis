from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.preprocessing import text
import torch
from torch.utils.data import Dataset
import numpy as np

# TODO: Early tweets only have 140 charachters?
MAX_SEQUENCE_LENGTH = 140
TOP_K = 20000

def sequence_vectorize(texts):
    """Vectorizes texts as sequence vectors.
    # Arguments
        train_texts: list, training text strings.
    # Returns
        x_train, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(texts)

    # Vectorize text.
    vectors = tokenizer.texts_to_sequences(texts)

    # Get max sequence length.
    max_length = len(max(vectors, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # Add padding to sequences.
    padded_vectors = sequence.pad_sequences(vectors, maxlen=max_length)

    return padded_vectors, tokenizer.word_index


# Data Vocabulary class
class SentAnaVocabulary(object):
  """Class to process text and extract vocabulary for mapping ,
       to make it easy to tokenize the words lookup with token or index of the token"""

  def __init__(self, token_to_idx=None, mask_token="<MASK>", add_unk=True, unk_token="<UNK>"):
    """Initialises every dictionary and tokens"""
    if token_to_idx is None:
        token_to_idx = {}
    self._token_to_idx = token_to_idx

    self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
    
    self._add_unk = add_unk
    self._unk_token = unk_token
    self._mask_token = mask_token
    
    self.mask_index = self.add_token(self._mask_token)
    self.unk_index = -1
    if add_unk:
        self.unk_index = self.add_token(unk_token) 
    
  def add_token(self, token):
    """Update mapping dicts based on the token."""
    if token in self._token_to_idx:
        index = self._token_to_idx[token]
    else:
        index = len(self._token_to_idx)
        self._token_to_idx[token] = index
        self._idx_to_token[index] = token
    return index
        
  # def get_embedding(self, token):
  #     """Add a list of tokens into the Vocabulary"""
  #     if token in self.Glove:
  #       token_embedding = self.Glove.get(token)
  #     else:
  #       token_embedding = float(0) * 25
  #     return token_embedding

  def lookup_token(self, token):
      """Retrieve the index associated with the token or the UNK index if token isn't present."""
      if self.unk_index >= 0:
          return self._token_to_idx.get(token, self.unk_index)
      else:
          return self._token_to_idx[token]

  def lookup_index(self, index):
      """Return the token associated with the index"""
      if index not in self._idx_to_token:
          raise KeyError("the index (%d) is not in the Vocabulary" % index)
      return self._idx_to_token[index]

  def __len__(self):
      return len(self._token_to_idx)

  def __str__(self):
      return "<Vocabulary(size=%d)>" % len(self)


## Sentence Vectorizer
class SentAnaVectorizer(object):
  """ The Vectorizer which coordinates the Vocabularies and puts them to use"""    
  def __init__(self, SA_vocab):
      """
      Assigns an Class Global Object for further vectorizing of data after getting initialised SentAnaVocabulary Object
          SA_vocab (Vocabulary) for mapping words to integers
      """
      self.SA_vocab = SA_vocab
      #self.Data_x = torch.tensor([])


  @classmethod
  def from_Textlist(cls, texts):
      """Instantiate the vectorizer from the dataset dataframe  """
      SA_vocab = SentAnaVocabulary()
      for single_sentence in texts:
        # replace the row.sentence.split() with sentencepiece
          for token in single_sentence.split(' '):
              SA_vocab.add_token(token)
          # self.Data_x[index] = torch.tensor([list(embedding)])
      return cls(SA_vocab)

  def vectorize(self, sentence, vector_length=-1):
    """   Function vectorizes the DataSet using the existing vocab  """

    indices = [self.SA_vocab.lookup_token(token) for token in sentence.split(' ')]
    if vector_length < 0:
        vector_length = len(indices)

    out_vector = np.zeros(vector_length, dtype=np.int64)
    out_vector[:len(indices)] = indices
    # replace it with mask_index or other
    out_vector[len(indices):] = self.SA_vocab.mask_index

    return out_vector


## SentenceDataet
class SentAnaDataset(Dataset):
  def __init__(self, texts, vectorizer,max_seq_len):
    """
    Initialises the Dataset and call the vocab and vectorizer
    """
    self.texts = texts
    self._vectorizer = vectorizer
    
    measure_len = lambda sentence: len(sentence.split(" "))
    self._max_seq_length = min(max(map(measure_len, texts)),max_seq_len)


  @classmethod
  def load_dataset(cls, texts):
    """Load dataset and make a new vectorizer from scratch"""
    return cls(texts, SentAnaVectorizer.from_Textlist(texts),128)

  def get_vectorizer(self):
    """ returns the vectorizer """
    return self._vectorizer
    
  def get_num_batches(self, batch_size):
    """Given a batch size, return the number of batches in the dataset"""
    return len(self) // batch_size


def sequence_tokenizer(texts):
    """Vectorizes texts as sequence vectors."""
    dataset = SentAnaDataset.load_dataset(texts)    
    vectorizer = dataset.get_vectorizer()
    #for single_sentence in texts:
    #    outputs.append(torch.tensor(vectorizer.vectorize(single_sentence,128)))
    #train_x = torch.stack(outputs)
    return vectorizer , vectorizer.SA_vocab._token_to_idx