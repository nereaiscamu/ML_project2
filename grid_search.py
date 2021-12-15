from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import collections
import matplotlib.pyplot as plt
import sys
sys.path.append('./data/')
from one_hot_encoding import get_dataset_one_hot
from multi_hot_encoding import get_dataset_multi_hot
from models.lstm_chord_models import LSTMChord, LSTMChordEmbedding, LSTMChordEmbedding_Multihot
from models.lstm_melody_models import LSTM_Multihot, LSTM_Multihot_MLP
from argparse import ArgumentParser
import pickle
import pdb
from train import train as train_model
import time


def grid_search():
    dataset = 1
    embed_sizes = [64, 96, 128, 192]
    layers = [1, 2, 3, 4]

    train_accs = np.empty((len(embed_sizes), len(layers)))
    val_accs = np.empty((len(embed_sizes), len(layers)))
    epochs_training = np.empty((len(embed_sizes), len(layers)))

    best_acc = 0
    best_embed_size = 0
    best_layers = 0

    file_name = "search_results/grid_search_" + str(int(time.time())) + ".txt"
    f = open(file_name, "w+")

    for i, l in enumerate(layers):
        for j, d in enumerate(embed_sizes):
            tr_acc, val_acc, epoch = train(dataset, d, l)
            train_accs[j, i] = tr_acc
            val_accs[j, i] = val_acc
            epochs_training[j, i] = epoch
            f.write('Dataset %d\tEmbed dim: %d\tLayers: %d\tTrain acc: %.2f\tVal acc: %.2f\tEpochs: %d\n' %(dataset, d, l, tr_acc, val_acc, epoch))
            print('Dataset %d\tEmbed dim: %d\tLayers: %d\tTrain acc: %.2f\tVal acc: %.2f\tEpochs: %d' %(dataset, d, l, tr_acc, val_acc, epoch))
            if val_acc > best_acc:
                best_acc = val_acc
                best_embed_size = d
                best_layers = l
    
    f.write('\nBest Validation accuracy: %.2f\tEmbed dim: %d\tLayers: %d' %(best_acc, best_embed_size, best_layers))
    f.close()
    print('\nBest Validation accuracy: %.2f\tEmbed dim: %d\tLayers: %d' %(best_acc, best_embed_size, best_layers))


def train(dataset, hidden_dim, layers):
    np.random.seed(42)

    parser = ArgumentParser(description='Train a model')
    parser.add_argument('--dataset', type=int,
                        default=8,
                        help='')
    parser.add_argument('--hidden-dim', type=int,
                        default=64,
                        help='')
    parser.add_argument('--lstm-layers', type=int,
                        default=2,
                        help='')
    parser.add_argument('--max-epochs', type=int,
                        default=200,
                        help='')
    parser.add_argument('--early-stopping', type=int,
                        default=15,
                        help='')
    parser.add_argument('--seed', type=int,
                        default=42,
                        help='')
    parser.add_argument('--use-saved-dataset', type=bool,
                        default=True,
                        help='')
    parser.add_argument('--save-path', type=str,
                        # required=True,
                        default=None,
                        help='')
    parser.add_argument('--load-path', type=str,
                        # required=True,
                        default=None,
                        help='')
    
    args = parser.parse_args()
    args.dataset = dataset
    args.hidden_dim = hidden_dim
    args.layers = layers
    return train_model(args)


grid_search()
