#%%

# importring libraries 
import os
from argparse import Namespace
from collections import Counter
import json
import re
import string
import numpy as np
import pandas as pd
import torch 
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from tqdm import tqdm_notebook
import nltk.data
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer, BertModel
#import sentencepiece as spm
from tqdm import tqdm
import unicodedata
import six
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import logging 
  
#Create and configure logger 
logging.basicConfig(filename="BERT_BILSTM.log", 
                    format='%(asctime)s %(message)s', 
                    filemode='w') 

#Creating an object 
logger=logging.getLogger() 
  
#Setting the threshold of logger to DEBUG 
logger.setLevel(logging.DEBUG) 

#%%

#%%
## Data Loading 

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
    pos_df = load_directory_data(os.path.join(directory, "train_pos.txt"))
    neg_df = load_directory_data(os.path.join(directory, "train_neg.txt"))
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)
  else :
    test_df = load_directory_data(os.path.join(directory, "test.txt"))
    test_df["polarity"] = -1
  return test_df



# Fetch and Process the dataset files.
def fetch_and_load_datasets():
  """ Initialises both train and test dataframes"""
  oldpwd=os.getcwd()
  os.chdir("../data/mst_first/")
  train_df = load_dataset(os.getcwd(),True)
  test_df = load_dataset(os.getcwd(),False)
  os.chdir(oldpwd)
  return train_df,test_df
  
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' , do_lower_case=True)

class SentimentRNN_WBERT(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, embedding_dim , hidden_dim, output_size, batch_size, n_layers, drop_prob=0.5,Device=False):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN_WBERT, self).__init__()
        self.device = Device
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        # embedding 
        #       
        self.embedding = BertModel.from_pretrained('bert-base-uncased')

        print('true')
        for param in nn.Sequential(*list(self.embedding.children())[:-1]).parameters():
            param.requires_grad = False

        # LSTM LAyer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,dropout=drop_prob, batch_first=True,bidirectional=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.decoder1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim,output_size)
        # self.hidden = self.init_hidden(batch_size)

        self.relu = nn.ReLU()

    def forward(self, input, masks, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = input.size(0)
        sentence_length = input.size(1) * torch.ones(batch_size)

        embeds = self.embedding(input,masks)


        #packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, sentence_length, batch_first=True) 
        lstm_out, hidden = self.lstm(embeds[0], hidden)  


        # lstm_fw = lstm_out[:, :, :self.hidden_dim]
        # lstm_bw = lstm_out[:, :, self.hidden_dim:]
        
        #Fetching the hidden state of Backward and Forward
        lstm_out = torch.cat((hidden[0][-2, :, :], hidden[0][-1, :, :]), dim=1)

        lstm_out = self.decoder1(lstm_out)

        lstm_out = self.relu(lstm_out)

        lstm_out = self.dropout(lstm_out)

        lstm_out = self.decoder2(lstm_out)

        return F.log_softmax(lstm_out,dim=1),hidden
 
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if(self.device):
          hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda())
        else:
          hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_())
        
        return hidden


#%%

MAX_LEN = 128
print("Loading dataset")
train_df , test_df = fetch_and_load_datasets()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' , do_lower_case=True)

outputs,outputs_masks = [],[]
for index,row in train_df.iterrows():
  encode_sent = tokenizer.encode_plus(
            text=row['sentence'],           # Preprocessed sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_tensors='pt',           # Return PyTorch tensor
            truncation=True,
            return_attention_mask=True      # Return attention mask
            )
  outputs.append(torch.reshape(encode_sent.get('input_ids'), (-1,)))
  outputs_masks.append(torch.reshape(encode_sent.get('attention_mask'), (-1,)))


test_outputs,test_outputs_masks = [],[]
for index,row in test_df.iterrows():
  encode_sent = tokenizer.encode_plus(
              text=row['sentence'],           # Preprocessed sentence
              add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
              max_length=MAX_LEN,             # Max length to truncate/pad
              pad_to_max_length=True,         # Pad sentence to max length
              return_tensors='pt',            # Return PyTorch tensor
              truncation=True,
              return_attention_mask=True      # Return attention mask
              )
  test_outputs.append(torch.reshape(encode_sent.get('input_ids'), (-1,)))
  test_outputs_masks.append(torch.reshape(encode_sent.get('attention_mask'), (-1,)))


#%%
train_x = torch.stack(outputs)
train_x_mask = torch.stack(outputs_masks)
train_y = torch.tensor(train_df['polarity'], dtype=torch.long)

test_x = torch.stack(test_outputs)
test_x_mask = torch.stack(test_outputs_masks)

train_data=TensorDataset(train_x,train_x_mask,train_y)

test_data=TensorDataset(test_x,test_x_mask)

TrainData, ValidationData = random_split(train_data,[int(0.9*len(train_data)),len(train_data) - int(0.9*len(train_data))])

HIDDEN_DIM = 256
EMB_DIM = 768
BATCH_SIZE= 64
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
EPOCHS =3
batch_size =64
#Device = False

# loss and optimization functions
learningrate=0.0001

cuda_available = True
# Check CUDA
if not torch.cuda.is_available():
    cuda_available = False

Device = torch.device("cuda" if cuda_available else "cpu")


model = SentimentRNN_WBERT(EMB_DIM, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE, N_LAYERS)
train_loader = DataLoader(dataset=TrainData, batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=ValidationData, batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_data, batch_size=100)
loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
loss_func.to(Device)
model.train()


def count_parameters(model):
    trainable_param=sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_param=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameter:",trainable_param)
    print("Non Trainable parameter:",non_trainable_param)
    return 0

count_parameters(model)

#%%
# Model training 

epochs = 5 
clip=5 # gradient clipping

epoch_bar = tqdm(desc='Epochs',total=epochs,position=0)
train_bar = tqdm(desc='Training',total=len(train_loader),position=1,leave=True)
val_bar = tqdm(desc='Validation',total=len(valid_loader),position=2,leave=True)

# move model to GPU, if available
if(cuda_available):
    model.cuda()

epoch_bar.n=0
# train for some number of epochs
for e in range(epochs):
  epoch_bar.update()
  # batch loop
  model.train()
  train_bar.n = 0
  val_bar.n = 0
  running_loss = 0
  hidden = model.init_hidden(BATCH_SIZE)
  index = 0
  for inputs,masks, labels in train_loader:
    train_bar.update()
    if(cuda_available):
        inputs,masks, labels = inputs.cuda(), masks.cuda(), labels.cuda()

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    hidden = tuple([each.data for each in hidden])

    # --------------------------------------
    # step 1. zero the gradients
    model.zero_grad()

    # step 2. compute the output
    predictions,hidden  = model(inputs,masks,hidden)

    #print(predictions)
    loss = loss_func(predictions, labels)
    loss.backward(retain_graph=True)
    running_loss += (loss.detach()  - running_loss) / (index + 1)
    index+=1
    nn.utils.clip_grad_norm_(model.parameters(), clip)  
    optimizer.step()
    train_bar.set_postfix(loss=running_loss, epoch=e)
    # loss stats

  valid_hidden = model.init_hidden(BATCH_SIZE)
  val_losses = []
  model.eval()
  running_val_loss = 0
  index = 0
  for inputs, masks,labels in valid_loader:
    val_bar.update()
    if(cuda_available):
        inputs, masks, labels = inputs.cuda(), masks.cuda(), labels.cuda()

    valid_hidden = tuple([each.data for each in valid_hidden])

    output,valid_hidden = model(inputs,masks,valid_hidden)
            
    val_loss = loss_func(output, labels)
    running_val_loss += (val_loss.detach()  - running_val_loss) / (index + 1)
    index+=1
    val_losses.append(val_loss.item())
    val_bar.set_postfix(loss=val_loss, epoch=e)
      
  logger.info('train_loss '+str(running_loss)+'at epoch'+str(e+1))
  logger.info('Validation_loss '+str(running_val_loss)+'at epoch'+str(e+1))
  torch.save({
  'epoch': e+1,
  'model_state_dict': model.state_dict(),
  'optimizer_state_dict': optimizer.state_dict(),
  'train_loss': running_loss,
  'val_loss': running_val_loss
  }, 'entire_model_BERT_BiLSTM_'+str(e+1)+'.pt')  

model.eval()
test_hidden = model.init_hidden(100)
print(type(test_loader))

def classDefiner(x): 
    if x[0] > x[1]:
        return -1
    return 1

id=1

for inputs,masks in test_loader:
    if(cuda_available):
        inputs,masks = inputs.cuda() ,masks.cuda()

    test_hidden = tuple([each.data for each in test_hidden])

    test_predictions,test_hidden = model(inputs,masks,test_hidden)
    with open('output_bert_bilstm.txt','a+',encoding ="utf-8") as fp:
        op2 = test_predictions.cpu()
        preds = map(classDefiner,list(op2.detach().numpy()))
        for item in list(preds):
            fp.write("{},{}\n".format(id,item))
            id+=1

PATH = "entire_model_BERT_BILSTM.pt"

# Save
torch.save(model, PATH)

# %%



