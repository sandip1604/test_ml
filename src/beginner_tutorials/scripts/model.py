import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy as np
import sklearn
import torch

class model(nn.Module):
    def __init__(self, list_size, activation, dropout_prob):
        super(model,self).__init__()
        if activation == "ReLu":
            self.activation = nn.ReLU()
        if activation == "Tanh":
            self.activation = nn.Tanh()
        if activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "LeakyReLu":
            self.activation = nn.Leaky_ReLu()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)
        self.input_layer = nn.Linear(50,list_size[0])
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        for idx, size in enumerate(list_size):
            if idx < (len(list_size)-1):
                self.layers.append(nn.Linear(size,list_size[idx+1]))
        self.output_layer = nn.Linear(list_size[-1],2)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.dropout(self.activation(self.input_layer(x.type(torch.FloatTensor))))
        for layer in self.layers:

            x = self.dropout(self.activation(layer(x.type(torch.FloatTensor))))
        output = self.output_layer(x.type(torch.FloatTensor))
        return output
        
