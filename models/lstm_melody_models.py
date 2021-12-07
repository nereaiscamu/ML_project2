from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM_Multihot(nn.Module):
    def __init__(self, input_size, embed_size, lstm_hidden_size, target_size):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.target_size = target_size

        self.embed = nn.Linear(input_size, embed_size)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.lstm_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.output = nn.Linear(self.lstm_hidden_size, self.target_size)

    def forward(self, inputs, lengths):  
        embeddings = self.embed(inputs)
        embeddings = F.relu(embeddings)

        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        relu1 = F.relu(lstm_out)

        output = self.output(relu1)

        return output  # Sometimes you want a softmax here -- look at the loss documentation