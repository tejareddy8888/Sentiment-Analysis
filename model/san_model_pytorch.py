import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.utils.data import TensorDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from base_model import BaseModel
from embeddings import Loader
from embed import sequence_tokenizer
from metrics import accuracy, classDefiner
from tqdm import tqdm
import numpy as np
from transformers import BertTokenizer, BertModel


class SentimentSAN(nn.Module,BaseModel):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self,TrainData, ValidationData, embedding_type , embedding_dim,  hidden_dim, output_size, batch_size, n_layers, drop_prob=0.5,Device=torch.device("cpu")):
        """
        Initialize the model by setting up the layers.
        """
        super(SentimentSAN, self).__init__()
        self.device = Device
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.train_df = TrainData
        self.val_df = ValidationData
        self.loss_func = nn.NLLLoss()
        self.loss_func.to(self.device)

        # embedding 
        if self.embedding_type == 'GloVe':

            Glove = Loader().loadGloveModel() 

            self.tokenizer, word_index = sequence_tokenizer(TrainData['sentence'].tolist())

            weights_matrix = Loader().Create_EmbMatrix(Glove,word_index)

            def create_embedding_layer(weights_matrix):
              "Takes Embedding as matrix and load into the embedding layer"
              num_embeddings, embedding_dim = weights_matrix.shape
              emb_layer = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix),padding_idx=1)
              return emb_layer, num_embeddings, embedding_dim
      
            emb_layer , vocab_size, self.embedding_dim = create_embedding_layer(weights_matrix)
            self.embedding = emb_layer

        else :

            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' , do_lower_case=True)
            # embedding 
            #       
            self.embedding = BertModel.from_pretrained('bert-base-uncased')
            for param in nn.Sequential(*list(self.embedding.children())[:-1]).parameters():
                param.requires_grad = False

        # LSTM Layer
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, n_layers,dropout=drop_prob, batch_first=True,bidirectional=True)
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
        
        # linear and sigmoid layer
        self.decoder1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.decoder2 = nn.Linear(hidden_dim,output_size)
        # self.hidden = self.init_hidden(batch_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        #batch_size = x.size(0)
        #sentence_length = x.size(1) * torch.ones(batch_size)
        weight = next(self.parameters()).data

        if self.embedding_type == 'GloVe':
            input = self.preprocess_text(input)
            embeds = self.embedding(input)
            lstm_out, hidden = self.lstm(embeds, hidden) 
        else:
            input , masks = self.preprocess_text(input)
            embeds = self.embedding(input,masks)
            lstm_out, hidden = self.lstm(embeds[0], hidden) 

        lstm_fw = lstm_out[:, :, :self.hidden_dim]
        lstm_bw = lstm_out[:, :, self.hidden_dim:]
        
        #Fetching the hidden state of Backward and Forward
        H = lstm_fw + lstm_bw
        M = self.tanh(H)
        if(self.device!=torch.device("cpu")):
          W = weight.new(self.hidden_dim).normal_(std=0.1).cuda()
        else:
          W = weight.new(self.hidden_dim).normal_(std=0.1)
        W = W.reshape(-1,self.hidden_dim).t()
        intermediate_alpha = torch.matmul(M,W)

        alpha = self.softmax(intermediate_alpha)

        #alpha = alpha.reshape(alpha.size()[0],alpha.size()[1])

        R = torch.matmul(H.permute(0,2,1),alpha)
        R = torch.squeeze(R)

        h_star = self.tanh(R)

        h_drop = self.dropout(h_star)

        att_out = self.decoder2(h_drop)       

        return F.log_softmax(att_out,dim=1),hidden
 
    
    def init_hidden(self,batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data
        
        if(self.device!=torch.device("cpu")):
          hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda(),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_().cuda())
        else:
          hidden = (weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_(),
                   weight.new(self.n_layers*2, batch_size, self.hidden_dim).zero_())
        return hidden

    @classmethod
    def build(cls,TrainData, ValidationData,embedding_type , embedding_dim, hidden_dim, output_dim, batch_size, n_layers,dropout_prob,Device):
      cls.model = SentimentSAN(TrainData, ValidationData,embedding_type,embedding_dim, hidden_dim, output_dim, batch_size, n_layers,dropout_prob,Device)
      return cls.model

    def preprocess_text(self,input):
        if self.embedding_type == 'GloVe':
            outputs = []
            for single_sentence in input.tolist():
               outputs.append(torch.tensor(self.tokenizer.vectorize(single_sentence,128)))
            train_x = torch.stack(outputs)
            if(self.device!=torch.device("cpu")):
                return train_x.cuda()
            return train_x
        else:
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
            if(self.device!=torch.device("cpu")):
                return train_x.cuda(), train_x_mask.cuda()
            return train_x , train_x_mask

    def fit(self,epochs,learningrate,decayRate):
        self.epoch_bar = tqdm(desc='Epochs',total=epochs,position=0)
        self.train_bar = tqdm(desc='Training',total=round(self.train_df.shape[0]/self.batch_size),position=2,leave=True)
        self.val_bar = tqdm(desc='Validation',total=round(self.val_df.shape[0]/self.batch_size),position=3,leave=True)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learningrate)
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        if(self.device!=torch.device("cpu")):
            self.model.cuda()

        self.epoch_bar.n=0
        clip = 2
        # train for some number of epochs
        for e in range(epochs):

            self.epoch_bar.set_postfix(lr=learningrate)
            # batch loop
            self.model.train()
            self.train_bar.n = 0
            self.val_bar.n = 0
            running_loss = 0
            train_accuracy = 0
            hidden = self.model.init_hidden(self.batch_size)
            index = 0
            for itere in range(0,self.train_df.shape[0],self.batch_size):
                if self.train_df[itere:itere+self.batch_size].shape[0]%self.batch_size == 0:
                    inputs = self.train_df.loc[itere:itere+self.batch_size-1,'sentence']
                    labels = torch.Tensor(self.train_df.loc[itere:itere+self.batch_size-1,'polarity'].values).long()

                    if(self.device!=torch.device("cpu")):
                        labels = labels.cuda()
                    self.train_bar.update()

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    hidden = tuple([each.data for each in hidden])

                    # --------------------------------------
                    # step 1. zero the gradients
                    self.model.zero_grad()

                    # step 2. compute the output
                    predictions,hidden  = self.model(inputs,hidden)

                    #print(predictions.size())
                    loss = self.loss_func(predictions, labels)
                    loss.backward(retain_graph=True)
                    running_loss += (loss.detach()  - running_loss) / (index + 1)
                    train_accuracy += (accuracy(predictions,labels) - train_accuracy) / (index + 1)
                    index+=1
                    nn.utils.clip_grad_norm_(self.model.parameters(), clip)  
                    self.optimizer.step()
                    self.train_bar.set_postfix(loss=running_loss,acc=train_accuracy, epoch=e)
            # loss stats

            valid_hidden = self.model.init_hidden(self.batch_size)
            val_losses = []
            self.model.eval()
            running_val_loss = 0
            val_accuracy = 0
            index = 0
            for itere in range(0,self.val_df.shape[0],self.batch_size):
                if self.val_df[itere:itere+self.batch_size].shape[0]%self.batch_size == 0:
                    inputs = self.val_df[itere:itere+self.batch_size]['sentence']
                    labels = torch.Tensor(self.val_df.loc[itere:itere+self.batch_size-1,'polarity'].values).long()

                    if(self.device!=torch.device("cpu")):
                        labels = labels.cuda()
                    self.val_bar.update()
                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    valid_hidden = tuple([each.data for each in valid_hidden])

                    output,valid_hidden = self.model(inputs,valid_hidden)
                    
                    val_loss = self.loss_func(output, labels)
                    running_val_loss += (val_loss.detach()  - running_val_loss) / (index + 1)
                    val_accuracy += (accuracy(predictions,labels) - val_accuracy) / (index + 1)
                    index+=1
                    val_losses.append(val_loss.item())
                    self.val_bar.set_postfix(loss=val_loss,acc=val_accuracy,epoch=e)

            # Optimizer Learning Rate
            learningrate = self.optimizer.param_groups[0]['lr']   

            #LearningRate Scheduler 
            self.my_lr_scheduler.step()
            self.epoch_bar.update()
            self.save('saved_model/model_at_epoch_'+str(e)+'.pt')



    def predict(self,test_df):
        self.model.eval()
        test_hidden = self.model.init_hidden(100)

        id=1

        for itere in range(0,test_df.shape[0],100):
            inputs = test_df[itere:itere+100]['sentence']

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            test_hidden = tuple([each.data for each in test_hidden])

            test_predictions,test_hidden = self.model(inputs,test_hidden)
            with open('saved_model/output_bert_bilstm.txt','a+',encoding ="utf-8") as fp:
                op2 = test_predictions.cpu()
                preds = map(classDefiner,list(op2.detach().numpy()))
                for item in list(preds):
                    fp.write("{},{}\n".format(id,item))
            id+=1

    def save(self,PATH):
      torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
            }, PATH)

        

