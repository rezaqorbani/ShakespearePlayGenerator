# LSTM.py

import torch
import torch.nn as nn

'''The following parameters are provided to the net

Num-classes - is the number of output in this case it's the vocabulary size
Input size - Embedding size
Hidden layers, number of hidden layer in each cell, the more is better, but also will slow down the training
'''


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

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_size).zero_(),
            weight.new(self.num_layers * self.num_directions, batch_size, self.hidden_size).zero_()
        )
        return hidden
    
    ## Save the trained model
    def save_model(self, path):
        """Save a PyTorch model to a specified path."""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load_model(cls, path):
        """Load a PyTorch model from a specified path."""
        model = cls()  # Initialize the model

        model.load_state_dict(torch.load(path))
        model.eval()  # Set in evaluation mode
        return model