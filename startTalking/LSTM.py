# LSTM.py

import torch
import torch.nn as nn

'''The following parameters are provided to the net

Num-classes - is the number of output in this case 1
Input size -
Hidden layers, number of hidden layer in each cell, the more is better, but also will slow down the training
Num layers - we have one layer of LSTM (layer we will increase it)'''


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout=0, bidirectional= False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        
        # Input Layer (for embedding)
        self.embedding = nn.Embedding(output_size, input_size)  # output size is the vocabulary size, input size is the embedding size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size*self.num_directions, output_size)
 

            
    def forward(self, x, hidden):
        # Build Embeddings
        x=self.embedding(x)
        
        out, hidden = self.lstm(x, hidden) # hidden is (h_n, c_n)
        out = out.contiguous().view(-1, self.hidden_size * self.num_directions)

        out = self.fc(out)
        return out, hidden
