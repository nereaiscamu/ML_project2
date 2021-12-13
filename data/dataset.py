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

        # Encode CHORD from sequence to multi-hot
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
    Variable length dataset for multi-hot CHORD and MELODY embedding
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
        sequence = self.sequences[i].astype(int)
        melody = self.melody_encoding[i]
        target_sequence = self.target_sequence[i]
        sequence_length = sequence.shape[0]

        # Encode CHORD from sequence to multi-hot
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

        # Encode MELODY from sequence to embedding
        melody_encoded = np.zeros((self.max_length, 12))
        for i, pitch_sequence in enumerate(melody):
            for note in pitch_sequence:
                if note != -1:
                    melody_encoded[i, note] += 1
            if sum(melody_encoded[i]) > 0:
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


class MultiHot_MelodyBassEncoded_VLDataset(Dataset):
    '''
    Variable length dataset for multi-hot CHORD, MELODY embedding and BASS embedding
    '''
    def __init__(self, sequences, melody_encoding, bass_encoding, target_sequence, vocab_sizes):
        self.sequences = sequences
        self.melody_encoding = melody_encoding
        self.bass_encoding = bass_encoding
        self.target_sequence = target_sequence
        self.vocab_sizes = vocab_sizes
        self.num_one_hot = len(vocab_sizes)   # Number of one-hot vectors that form the multi-hot

        self.max_length = max(map(len, self.sequences))  # Add max length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        sequence = self.sequences[i]
        melody = self.melody_encoding[i]
        bass = self.bass_encoding[i]
        target_sequence = self.target_sequence[i]
        sequence_length = sequence.shape[0]

        # Encode CHORD from sequence to multi-hot
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

        # Encode MELODY from sequence to embedding
        melody_encoded = np.zeros((self.max_length, 12))
        for i, pitch_sequence in enumerate(melody):
            for note in pitch_sequence:
                if note != -1:
                    melody_encoded[i, note] += 1
            if sum(melody_encoded[i]) > 0:
                melody_encoded[i] /= sum(melody_encoded[i])

        # Encode BASS from sequence to embedding
        bass_encoded = np.zeros((self.max_length, 12))
        for i, bass_sequence in enumerate(bass):
            for note in bass_sequence:
                bass_encoded[i, note] += 1
            bass_encoded[i] /= sum(bass_encoded[i])
            
        input_ = np.concatenate((encoded_multihot, melody_encoded), axis=1)
        input_ = np.concatenate((input_, bass_encoded), axis=1)

        # Pad target with some INVALID value (-1)
        target = np.ones(self.max_length - 1) * -1
        target[:sequence_length - 1] = target_sequence[1:]
        return {
            "input": torch.tensor(input_[:-1]),
            "target": torch.tensor(target).long(),
            "length": sequence_length - 1,  # Return the length
        }


class MultiHot_MelodyDurationEncoded_VLDataset(Dataset):
    '''
    Variable length dataset for multi-hot CHORD and MELODY embedding
    '''
    def __init__(self, sequences, melody_encoding, target_sequence, vocab_sizes):
        self.sequences = sequences
        self.melody_encoding = melody_encoding # Here melody_encoding should be a array of tuples of (melody_sequence, duration_sequence)
        self.target_sequence = target_sequence
        self.vocab_sizes = vocab_sizes
        self.num_one_hot = len(vocab_sizes)   # Number of one-hot vectors that form the multi-hot

        self.max_length = max(map(len, self.sequences))  # Add max length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, i):
        sequence = self.sequences[i].astype(int)
        melody = self.melody_encoding[i]
        target_sequence = self.target_sequence[i]
        sequence_length = sequence.shape[0]

        # Encode CHORD from sequence to multi-hot
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

        # Encode MELODY from sequence to embedding
        melody_encoded = np.zeros((self.max_length, 12))

        # Extract tuple
        tuple = melody[0]
        (pitch_sequences, duration_sequences) = tuple

        for i, (pitch_seq, duration_seq) in enumerate(zip(pitch_sequences, duration_sequences)):
            # take only last 10 notes
            #if len(pitch_seq) > 10:
            #    pitch_seq = pitch_seq[-10:]
            #    duration_seq = duration_seq[-10:]

            duration_seq = np.array(duration_seq)
            duration_seq = np.where(duration_seq < 0, 0, duration_seq)

            #Create weighted array
            divider = np.arange(float(2*len(pitch_seq)), step=2)
            divider[0] = 1.0
            divider = divider[::-1]

            # TODO series gets too big if we take all the notes
            #divider = np.geomspace(1, np.power(2, len(pitch_seq)-1), num=len(pitch_seq))
            #divider = np.flipud(divider)

            melody_encoded[i, pitch_seq] += duration_seq / divider
            
            if sum(melody_encoded[i]) > 0:
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


class MultiHot_MelodyWeighted_VLDataset(Dataset):
    '''
    Variable length dataset for multi-hot CHORD and MELODY embedding
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
        sequence = self.sequences[i].astype(int)
        melody = self.melody_encoding[i]
        target_sequence = self.target_sequence[i]
        sequence_length = sequence.shape[0]

        # Encode CHORD from sequence to multi-hot
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

        # Encode MELODY from sequence to embedding
        melody_encoded = np.zeros((self.max_length, 12))
        for i, pitch_sequence in enumerate(melody):
            # take only last 10 notes
            if len(pitch_sequence) > 10:
                pitch_sequence = pitch_sequence[-10:]

            # Create geometric series
            divider = np.geomspace(1, np.power(2, len(pitch_sequence)-1), num=len(pitch_sequence))
            # Flip series
            divider = np.flipud(divider)

            for idx_div, note in enumerate(pitch_sequence):
                if note != -1:
                    melody_encoded[i, note] += divider[idx_div]

            if sum(melody_encoded[i]) > 0:
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