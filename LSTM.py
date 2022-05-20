"""
LSTM Model for Yelp Sentiment Analysis
Base code provided by Huajie Shao, Ph.D
Code completed by Roger Clanton, David Ni, and Evan Ward
"""

import torch
from torch import nn
import pandas as pd 
import numpy as np
import torch.autograd as autograd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSTMModel(nn.Module):
    """
    LSTM model that will be used to perform Sentiment analysis.
    """
    def __init__(self, vocab_size, output_size, embedding_dim, embedding_matrix,\
        hidden_dim, n_layers, input_len, pretrain=False):
        """
        Initialize the model by setting up the layers.
        """
        super().__init__()
        
        self.output_size = output_size  # y_out size = 1
        self.n_layers = n_layers   # layers of LSTM
        self.hidden_dim = hidden_dim  # hidden dim of LSTM
        self.input_len = input_len # len of input features
        
        ## set up pre-train embeddings. if true, load pretrain-embedding from GloVe
        if pretrain:
            # print("import glove embedding to nn.Embedding now")
            self.embedding = nn.Embedding.from_pretrained(embedding_matrix,freeze=False)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.init_weights()
        
        ## define LSTM model
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        ## dropout layer
        self.dropout = nn.Dropout(0.2)
        
        ## max pool
        self.pool = nn.MaxPool1d(self.input_len)
        
        ## linear and sigmoid layers
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sigmoid = nn.Sigmoid()
        

    ## initialize the weights
    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
    
    ## initial hidden state and cell state
    def _init_hidden(self, batch_size):
        return(autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)).to(device),
                autograd.Variable(torch.randn(self.n_layers, batch_size, self.hidden_dim)).to(device)
                )

    
    ## feed input x into LSTM model for training/testing
    def forward(self, x):
        batch_size = x.size(0)

        ## hidden_cell=(hidden, cell) where hidden=layer * batch * dim
        hidden_cell = self._init_hidden(batch_size)
        #feed input sequence x to get embeddings
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds,hidden_cell)

        # permute the position of hidden_dim and seq_len
        lstm_out = lstm_out.permute(0,2,1)

        # pool lstm layer output to reduce seq_len dim-
        out = self.pool(lstm_out)
        out = out.view(out.size(0),-1)
        
        ## dropout to prevent overfitting
        out = self.dropout(out)

        ## feed into linear layer
        out = self.fc(out)
        out = self.sigmoid(out)
        out = out[:,0]
        
        return out
        