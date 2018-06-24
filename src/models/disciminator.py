import torch.nn as nn
import math
import ref
from LSTM import GraphLSTM

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gLSTM = GraphLSTM(input_size, hidden_size)
        self.out = nn.Linear(16*input_size, 1),

    def forward(self, features):
        """
        Args:
        features: [batch_size, 16, 3]
        """
        gLSTM_output = self.gLSTM(features)
        flatten_gLSTM = gLSTM_output.view(-1,16*input_size)
        output = self.out(flatten_gLSTM)
        return output
