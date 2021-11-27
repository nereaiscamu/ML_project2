'''
Mock dataset from Andrew's tutorial
'''
import torch
from torch.utils.data import Dataset
import numpy as np
import pdb

class VLDataset(Dataset):
  def __init__(self, sequences, vocab_size):
    self.sequences = sequences
    self.vocab_size = vocab_size

    self.max_length = max(map(len, self.sequences))  # Add max length

  def __len__(self):
    return len(self.sequences)

  def __getitem__(self, i):
    sequence = self.sequences[i]
    sequence_length = len(self.sequences[i])

    # Pad input with 0 up to max length
    encoded_sequence = np.zeros((self.max_length, self.vocab_size))
    encoded_sequence[np.arange(sequence_length), sequence] = 1

    # Pad target with some INVALID value (-1)
    target = np.ones(self.max_length - 1) * -1
    target[:sequence_length - 1] = sequence[1:]

    return {
        "input": torch.tensor(encoded_sequence[:-1]),
        "target": torch.tensor(target).long(),
        "length": sequence_length - 1,  # Return the length
    }


class MultiHot_VLDataset(Dataset):
    def __init__(self, sequences, target_sequence, vocab_sizes):
      self.sequences = sequences
      self.target_sequence = target_sequence
      self.vocab_sizes = vocab_sizes
      self.num_one_hot = len(vocab_sizes)   # Number of one-hot vectors that form the multi-hot

      self.max_length = max(map(len, self.sequences[0]))  # Add max length

    def __len__(self):
      return len(self.sequences)

    def __getitem__(self, i):
      sequence = self.sequences[i]
      target_sequence = self.target_sequence[i]
      sequence_length = len(sequence[0])

      # Pad input with 0 up to max length
      encoded_sequence = np.zeros((self.max_length, sum(self.vocab_sizes)))
      for n in range(self.num_one_hot):
        encoded_sequence[np.arange(sequence_length), sequence[n]] = 1 # FIXME need to transform into one-hot and then concat
        
      # Pad target with some INVALID value (-1)
      target = np.ones(self.max_length - 1) * -1
      pdb.set_trace()
      target[:sequence_length - 1] = target_sequence[1:]

      return {
          "input": torch.tensor(encoded_sequence[:-1]),
          "target": torch.tensor(target).long(),
          "length": sequence_length - 1,  # Return the length
      }

def get_mock_dataset(test_split=0.2):
    vocab_size = 5
    num_sequences = 20
    max_sequence_length = 40

    sequences = []
    for _ in range(num_sequences):
        sequence_length = np.random.randint(max_sequence_length - 2) + 2
        sequences.append(np.ones((sequence_length), dtype=int) * np.random.randint(vocab_size))

    # Split
    # TODO make split random
    split_idx = int(num_sequences*(1-test_split))
    train_seq = sequences[:split_idx]
    test_seq = sequences[split_idx:]

    train_dataset = VLDataset(train_seq, vocab_size)
    test_dataset = VLDataset(test_seq, vocab_size)
    return train_dataset, test_dataset, vocab_size