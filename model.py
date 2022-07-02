import torch
import torch.nn as nn
import numpy as np

# basic sequential block
class Block(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm = True,
                 dropout_p=.4):
    
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p

        super().__init__()

        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)

        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size)
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        
        # |y| = (batch_size, output_size)
        return y.squeeze()

class MLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[32, 16, 8, 4], 
                 use_batch_norm=True,
                 dropout_p=.4):

        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"

        last_hidden_size = input_size
        hidden_layers = []
        for hidden_size in hidden_sizes:
            hidden_layers += [Block(
                            last_hidden_size, 
                            hidden_size, 
                            use_batch_norm, 
                            dropout_p)]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *hidden_layers,
            nn.Linear(last_hidden_size, output_size),
        )

    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.layers(x)

        # |y| = (batch_size, output_size)
        return y.squeeze()

class LSTM(nn.Module):
    def __init__(self, 
                input_size, 
                output_size,
                lstm_hidden_size=32, 
                hidden_sizes=[16, 8, 4],
                use_batch_norm=True,
                n_layers=2, 
                dropout_p = 0.3) -> None:

        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            num_layers=n_layers,
        )

        last_hidden_size = lstm_hidden_size
        hidden_layers = []
        for hidden_size in hidden_sizes:
            hidden_layers += [Block(
                            last_hidden_size, 
                            hidden_size, 
                            use_batch_norm, 
                            dropout_p)]
            last_hidden_size = hidden_size

        self.layers = nn.Sequential(
            *hidden_layers,
            nn.Linear(last_hidden_size, output_size),
        )
    
    def forward(self, x):
        y_rnn , _ = self.rnn(x)
        y = self.layers(y_rnn)
        return y
