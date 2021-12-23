import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM_Multihot(nn.Module):
    def __init__(self, input_size, embed_size, lstm_hidden_size, target_size, num_layers=1, dropout_linear=0, dropout_lstm=0):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.target_size = target_size
        self.num_layers = num_layers

        self.embed = nn.Linear(input_size, embed_size)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.lstm_hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout_lstm        # if non-zero, introduces dropout of outputs of every lstm layer except the last out
        )

        self.dropout_linear = nn.Dropout(p=dropout_linear)
        self.output = nn.Linear(self.lstm_hidden_size, self.target_size)

    def forward(self, inputs, lengths):  
        embeddings = F.relu(self.embed(inputs))

        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        out = F.relu(lstm_out)

        out = self.dropout_linear(out)
        out = self.output(out)

        return out 

class LSTM_Multihot_MLP(nn.Module):
    def __init__(self, input_size, embed_size, lstm_hidden_size, target_size, num_layers=1, dropout_linear=0, dropout_lstm=0):
        super().__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size
        self.target_size = target_size
        self.num_layers = num_layers

        self.embed = nn.Linear(input_size, embed_size)
        self.embed2 = nn.Linear(embed_size, embed_size)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.lstm_hidden_size,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True,
            dropout=dropout_lstm        # if non-zero, introduces dropout of outputs of every lstm layer except the last out
        )

        self.dropout_linear = nn.Dropout(p=dropout_linear)
        self.hidden = nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size)
        self.output = nn.Linear(self.lstm_hidden_size, self.target_size)

    def forward(self, inputs, lengths):  
        embeddings = F.relu(self.embed(inputs))
        embeddings = F.relu(self.embed2(embeddings))

        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        out = F.relu(lstm_out)

        out = self.dropout_linear(out)
        out = F.relu(self.hidden(out))
        out = self.dropout_linear(out)
        out = self.output(out)

        return out