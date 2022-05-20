"""
Data loader for LSTM model
Base code provided by Huajie Shao, Ph.D
Feature and label loading completed by Roger Clanton, David Ni, and Evan Ward
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
from ast import literal_eval


'''Create data loader'''

class MovieDataset(Dataset):
    def __init__(self, filename):
        # Line caused error :(
        # self.df = pd.read_csv(filename, converters={'input_x': literal_eval})
        self.df = pd.read_csv(filename)
        # print(self.df['input_x'])

    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # load the input features and labels
        input_x = self.df.loc[index,'input_x']
        input_x = input_x[1:-1]
        # convert features into list for easier conversion to tensor later
        input_x = list(map(int,input_x.split(", ")))
        label = self.df.loc[index,'Label']

        return torch.tensor(input_x), torch.tensor(label,dtype=torch.float)
        
        