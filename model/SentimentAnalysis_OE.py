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
#import tensorflow
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from sklearn.model_selection import train_test_split
import logging
from transformers import BertTokenizer, BertModel

#Create and configure logger
logging.basicConfig(filename="checkpoint_attention/BERT_BILSTM_attention.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

#Creating an object
logger=logging.getLogger()

#Setting the threshold of logger to DEBUG
logger.setLevel(logging.DEBUG)

#tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased' , do_lower_case=True)

# embedding = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
#%%

#%%
## Data Loading

def load_directory_data(file):
  """Retrieves the Sentences from the input text file into a Dict,stores and return into Dataframe """
  data = {}
  data["sentence"] = []
  num_lines = sum(1 for line in open(file,'r'))
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


def load_Pretraindataset():
  """ Specifically for assigning labels and get the train_df or test_df from the pre-defined datasets"""
  pos_df = load_directory_data("../data/pretrain_data/train_pos.txt")
  neg_df = load_directory_data("../data/pretrain_data/train_neg.txt")
  pos_df["polarity"] = 1
  neg_df["polarity"] = 0
  return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)





# Fetch and Process the dataset files.
def fetch_and_load_datasets():
  """ Initialises both train and test dataframes"""
  oldpwd=os.getcwd()
  os.chdir("../data/pipeline_elect/")
  train_df = load_dataset(os.getcwd(),True)
  test_df = load_dataset(os.getcwd(),False)
  os.chdir(oldpwd)
  return train_df,test_df



class SentimentRNN_WBERT(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, embedding_dim , hidden_dim, output_size, batch_size, n_layers, drop_prob=0.5,Device=True):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentRNN_WBERT, self).__init__()
        self.device = Device
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' , do_lower_case=True)
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

    def preprocess_text(self,input):
        outputs,outputs_masks = [],[]
        for index,value in input.items():
          encode_sent = self.tokenizer.encode_plus(
                    text=value,           # Preprocessed sentence
                    add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
                    max_length=140,                  # Max length to truncate/pad
                    pad_to_max_length=True,         # Pad sentence to max length
                    return_tensors='pt',           # Return PyTorch tensor
                    truncation=True,
                    return_attention_mask=True      # Return attention mask
                    )
          outputs.append(torch.reshape(encode_sent.get('input_ids'), (-1,)))
          outputs_masks.append(torch.reshape(encode_sent.get('attention_mask'), (-1,)))
        train_x = torch.stack(outputs)
        train_x_mask = torch.stack(outputs_masks)
        if(self.device):
            return train_x.cuda(), train_x_mask.cuda()
        return train_x , train_x_mask

    def forward(self, input, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        input , masks = self.preprocess_text(input)

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




MAX_LEN = 40
print("Loading dataset")
train_df , test_df = fetch_and_load_datasets()
train_df, val_df = train_test_split(train_df,shuffle=True,train_size=0.9,test_size=0.1,random_state=42)

pretrain_df = load_Pretraindataset()

pretrain_df.reset_index(drop=True,inplace=True)
train_df.reset_index(drop=True,inplace=True)
val_df.reset_index(drop=True,inplace=True)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' , do_lower_case=True)

#%%
HIDDEN_DIM = 300
EMB_DIM = 768
BATCH_SIZE=16
OUTPUT_DIM = 2
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
#Device = False

# loss and optimization functions
learningrate=0.0001

cuda_available = True
# Check CUDA
if not torch.cuda.is_available():
    cuda_available = False

Device = torch.device("cuda" if cuda_available else "cpu")


#%%
model = SentimentRNN_WBERT(EMB_DIM, HIDDEN_DIM, OUTPUT_DIM, BATCH_SIZE, N_LAYERS,DROPOUT,cuda_available)

loss_func = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learningrate)
my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
loss_func.to(Device)


model.train()


def count_parameters(model):
    trainable_param=sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_param=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print("Trainable parameter:",trainable_param)
    print("Non Trainable parameter:",non_trainable_param)
    return 0

count_parameters(model)

def classDefiner_accuracy(x):
    if x[0] > x[1]:
        return 0
    return 1

def accuracy(predicted,labels):
    acc = 0
    preds = map(classDefiner_accuracy,list(predicted.cpu().detach().numpy()))
    labels = list(labels.cpu().detach().numpy())
    preds = list(preds)
    for i in range(0,len(preds)):
      if preds[i] == labels[i]:
        acc+=1
    return acc/len(preds)


#%%
# Model training

epochs = 6
clip=2 # gradient clipping

epoch_bar = tqdm(desc='Epochs',total=epochs,position=0)
pretrain_bar = tqdm(desc='Pretraining',total=round(pretrain_df.shape[0]/BATCH_SIZE),position=1)
train_bar = tqdm(desc='Training',total=round(train_df.shape[0]/BATCH_SIZE),position=2,leave=True)
val_bar = tqdm(desc='Validation',total=round(val_df.shape[0]/BATCH_SIZE),position=3,leave=True)


# move model to GPU, if available
if(cuda_available):
    model.cuda()

model.train()

for e in range(2):
    running_loss = 0
    train_accuracy = 0
    pretrain_bar.n = 0
    hidden = model.init_hidden(BATCH_SIZE)
    index = 0
    for itere in range(0,pretrain_df.shape[0],BATCH_SIZE):
        if pretrain_df[itere:itere+BATCH_SIZE].shape[0]%BATCH_SIZE == 0:
            inputs = pretrain_df.loc[itere:itere+BATCH_SIZE-1,'sentence']
            labels = torch.Tensor(pretrain_df.loc[itere:itere+BATCH_SIZE-1,'polarity'].values).long()
            if(cuda_available):
                labels = labels.cuda()
            pretrain_bar.update()
            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            hidden = tuple([each.data for each in hidden])

            # --------------------------------------
            # step 1. zero the gradients
            model.zero_grad()

            # step 2. compute the output
            predictions,hidden  = model(inputs,hidden)

            #print(predictions.size())
            loss = loss_func(predictions, labels)
            loss.backward(retain_graph=True)
            running_loss += (loss.detach()  - running_loss) / (index + 1)
            index+=1
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            pretrain_bar.set_postfix(loss=running_loss, epoch=e)

PATH = "checkpoint_attention/entire_model_BERT_BILSTM_pretrain.pt"
# Save
torch.save(model, PATH)



epoch_bar.n=0
# train for some number of epochs
for e in range(epochs):
    epoch_bar.update()
    epoch_bar.set_postfix(lr=learningrate)
    # batch loop
    model.train()
    train_bar.n = 0
    val_bar.n = 0
    running_loss = 0
    train_accuracy = 0
    hidden = model.init_hidden(BATCH_SIZE)
    index = 0
    for itere in range(0,train_df.shape[0],BATCH_SIZE):
        if train_df[itere:itere+BATCH_SIZE].shape[0]%BATCH_SIZE == 0:
            inputs = train_df.loc[itere:itere+BATCH_SIZE-1,'sentence']
            labels = torch.Tensor(train_df.loc[itere:itere+BATCH_SIZE-1,'polarity'].values).long()
            if(cuda_available):
                labels = labels.cuda()
            train_bar.update()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            hidden = tuple([each.data for each in hidden])

            # --------------------------------------
            # step 1. zero the gradients
            model.zero_grad()

            # step 2. compute the output
            predictions,hidden  = model(inputs,hidden)

            #print(predictions.size())
            loss = loss_func(predictions, labels)
            loss.backward(retain_graph=True)
            running_loss += (loss.detach()  - running_loss) / (index + 1)
            train_accuracy += (accuracy(predictions,labels) - train_accuracy) / (index + 1)
            index+=1
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            train_bar.set_postfix(loss=running_loss,acc=train_accuracy, epoch=e)
    # loss stats

    valid_hidden = model.init_hidden(BATCH_SIZE)
    val_losses = []
    model.eval()
    running_val_loss = 0
    val_accuracy = 0
    index = 0
    for itere in range(0,val_df.shape[0],BATCH_SIZE):
        if val_df[itere:itere+BATCH_SIZE].shape[0]%BATCH_SIZE == 0:
            inputs = val_df[itere:itere+BATCH_SIZE]['sentence']
            labels = torch.Tensor(val_df.loc[itere:itere+BATCH_SIZE-1,'polarity'].values).long()

            if(cuda_available):
                labels = labels.cuda()
            val_bar.update()

            valid_hidden = tuple([each.data for each in valid_hidden])

            output,valid_hidden = model(inputs,valid_hidden)

            val_loss = loss_func(output, labels)
            running_val_loss += (val_loss.detach()  - running_val_loss) / (index + 1)
            val_accuracy += (accuracy(predictions,labels) - val_accuracy) / (index + 1)
            index+=1
            val_losses.append(val_loss.item())
            val_bar.set_postfix(loss=val_loss,acc=val_accuracy,epoch=e)

    # Optimizer Learning Rate
    learningrate = optimizer.param_groups[0]['lr']

    #LearningRate Scheduler
    my_lr_scheduler.step()

    logger.info('train_loss '+str(running_loss)+'at epoch'+str(e+1))
    logger.info('Validation_loss '+str(running_val_loss)+'at epoch'+str(e+1))
    torch.save({
    'epoch': e+1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'train_loss': running_loss,
    'val_loss': running_val_loss
    }, 'checkpoint_attention/entire_model_BERT_BiLSTM'+str(e+1)+'.pt')

model.eval()
test_hidden = model.init_hidden(100)

def classDefiner(x):
    if x[0] > x[1]:
        return -1
    return 1

id=1

for itere in range(0,test_df.shape[0],100):
    inputs = test_df[itere:itere+100]['sentence']
    test_hidden = tuple([each.data for each in test_hidden])

    test_predictions,test_hidden = model(inputs,test_hidden)
    with open('checkpoint_attention/output_bert_bilstm.txt','a+',encoding ="utf-8") as fp:
        op2 = test_predictions.cpu()
        preds = map(classDefiner,list(op2.detach().numpy()))
        for item in list(preds):
            fp.write("{},{}\n".format(id,item))
            id+=1

PATH = "checkpoint_attention/entire_model_BERT_BILSTM_att.pt"

# Save
torch.save(model, PATH)

# %%
