import torch
import torch.nn.functional as F
from .lora import LoraLinear
from functools import partial

# MLP Model
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, dropout = 0.5, rank = 0):
        
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        
        self.linear_layers = torch.nn.ModuleList()

        linear_layer = torch.nn.Linear if rank == 0 else partial(LoraLinear, r=rank, lora_dropout=dropout)
            
        # Adding input layer
        self.linear_layers.append(linear_layer(in_channels, hidden_channels))
            
        # Adding hidden layers
        for i in range(num_layers-2):
            self.linear_layers.append(linear_layer(hidden_channels, hidden_channels))
        
        # Adding output layer
        self.linear_layers.append(linear_layer(hidden_channels, out_channels))
    
    def forward(self, x):
        for i in range(self.num_layers-1): # exclude output 
            x = self.linear_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # transformation phase (output layer)
        x = self.linear_layers[-1](x)

        x = F.log_softmax(x, dim=1)

        return x