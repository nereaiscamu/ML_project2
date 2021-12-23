import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMChord(nn.Module):
    def __init__(self, vocab_size, lstm_hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(
            self.vocab_size,
            self.lstm_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.output = nn.Linear(self.lstm_hidden_size, self.vocab_size)

    def forward(self, inputs, lengths):  # Add lengths to input
        # Shape = batch, sequence, vocab_size
        packed = pack_padded_sequence(inputs, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        relu1 = F.relu(lstm_out)

        # Shape = batch, sequence, lstm_hidden_size
        output = self.output(relu1)

        # Shape = batch, sequence, vocab_size
        return output  # Sometimes you want a softmax here -- look at the loss documentation

class LSTMChordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, lstm_hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.lstm_hidden_size = lstm_hidden_size

        self.embed = nn.Linear(vocab_size, embed_size)

        self.lstm = nn.LSTM(
            self.embed_size,
            self.lstm_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True,
        )

        self.output = nn.Linear(self.lstm_hidden_size, self.vocab_size)

    def forward(self, inputs, lengths):  # Add lengths to input
        # Shape = batch, sequence, vocab_size
        embeddings = self.embed(inputs)
        embeddings = F.relu(embeddings)

        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        relu1 = F.relu(lstm_out)

        # Shape = batch, sequence, lstm_hidden_size
        output = self.output(relu1)

        # Shape = batch, sequence, vocab_size
        return output  # Sometimes you want a softmax here -- look at the loss documentation


class LSTMChordEmbedding_Multihot(nn.Module):
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

    def forward(self, inputs, lengths):  # Add lengths to input
        # Shape = batch, sequence, vocab_size
        embeddings = self.embed(inputs)
        embeddings = F.relu(embeddings)

        packed = pack_padded_sequence(embeddings, lengths, enforce_sorted=False, batch_first=True)
        lstm_out_packed, hidden_out = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        relu1 = F.relu(lstm_out)

        # Shape = batch, sequence, lstm_hidden_size
        output = self.output(relu1)

        # Shape = batch, sequence, vocab_size
        return output  # Sometimes you want a softmax here -- look at the loss documentation

