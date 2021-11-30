'''
Mock dataset from Andrew's tutorial
'''
import torch
from torch.utils.data import Dataset
import numpy as np
import pdb

class OneHot_VLDataset(Dataset):
    '''
    Variable length dataset for CHORD encoding as a ONE-HOT
    Author: Andrew
    '''
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
    '''
    Variable length dataset for CHORD encoding as a MULTI-HOT
    '''
    def __init__(self, sequences, target_sequence, vocab_sizes):
        self.sequences = sequences
        self.target_sequence = target_sequence
        self.vocab_sizes = vocab_sizes
        self.num_one_hot = len(vocab_sizes)   # Number of one-hot vectors that form the multi-hot

        self.max_length = max(map(len, self.sequences))  # Add max length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        sequence = self.sequences[i]
        target_sequence = self.target_sequence[i]
        sequence_length = sequence.shape[0]

        # Pad input with 0 up to max length
        encoded_multihot = None
        for n in range(self.num_one_hot):
            if encoded_multihot is None:
                encoded_onehot = np.zeros((self.max_length, self.vocab_sizes[n]))
                encoded_onehot[np.arange(sequence_length), sequence[:,n]] = 1
                encoded_multihot = encoded_onehot

            else:
                encoded_onehot = np.zeros((self.max_length, self.vocab_sizes[n]))
                encoded_onehot[np.arange(sequence_length), sequence[:,n]] = 1 
                encoded_multihot = np.concatenate((encoded_multihot, encoded_onehot), axis=1)

        
        # Pad target with some INVALID value (-1)
        target = np.ones(self.max_length - 1) * -1
        target[:sequence_length - 1] = target_sequence[1:]
        return {
            "input": torch.tensor(encoded_multihot[:-1]),
            "target": torch.tensor(target).long(),
            "length": sequence_length - 1,  # Return the length
        }


class MultiHot_MelodyEncoded_VLDataset(Dataset):
    '''
    Variable length dataset for CHORD encoding as a MULTI-HOT
    '''
    def __init__(self, sequences, melody_encoding, target_sequence, vocab_sizes):
        self.sequences = sequences
        self.melody_encoding = melody_encoding
        self.target_sequence = target_sequence
        self.vocab_sizes = vocab_sizes
        self.num_one_hot = len(vocab_sizes)   # Number of one-hot vectors that form the multi-hot

        self.max_length = max(map(len, self.sequences))  # Add max length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        sequence = self.sequences[i]
        melody = self.melody_encoding[i]
        target_sequence = self.target_sequence[i]
        sequence_length = sequence.shape[0]

        # Pad input with 0 up to max length
        encoded_multihot = None
        for n in range(self.num_one_hot):
            if encoded_multihot is None:
                encoded_onehot = np.zeros((self.max_length, self.vocab_sizes[n]))
                encoded_onehot[np.arange(sequence_length), sequence[:,n]] = 1
                encoded_multihot = encoded_onehot

            else:
                encoded_onehot = np.zeros((self.max_length, self.vocab_sizes[n]))
                encoded_onehot[np.arange(sequence_length), sequence[:,n]] = 1 
                encoded_multihot = np.concatenate((encoded_multihot, encoded_onehot), axis=1)


        melody_encoded = np.zeros((self.max_length, 12))
        for i, pitch_sequence in enumerate(melody):
            for note in pitch_sequence:
                melody_encoded[i, note] += 1
            melody_encoded[i] /= sum(melody_encoded[i])
        input_ = np.concatenate((encoded_multihot, melody_encoded), axis=1)
        # Pad target with some INVALID value (-1)
        target = np.ones(self.max_length - 1) * -1
        target[:sequence_length - 1] = target_sequence[1:]
        return {
            "input": torch.tensor(input_[:-1]),
            "target": torch.tensor(target).long(),
            "length": sequence_length - 1,  # Return the length
        }
