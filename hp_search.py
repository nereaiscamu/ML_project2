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
    dataset = 5
    #embed_sizes = [64, 96, 128, 192]
    embed_sizes = [128, 192]
    layers = [1, 2, 3, 4]

    train_accs = np.empty((len(embed_sizes), len(layers)))
    val_accs = np.empty((len(embed_sizes), len(layers)))
    epochs_training = np.empty((len(embed_sizes), len(layers)))

    best_acc = 0
    best_embed_size = 0
    best_layers = 0

    file_name = "search_results/grid_search_" + str(int(time.time())) + ".txt"
    f = open(file_name, "a")
    f.write('Grid search results:\n')
    f.close()

    for i, l in enumerate(layers):
        for j, d in enumerate(embed_sizes):
            tr_acc, val_acc, epoch = train(dataset, d, l)
            train_accs[j, i] = tr_acc
            val_accs[j, i] = val_acc
            epochs_training[j, i] = epoch
            f = open(file_name, "a")
            f.write('Dataset %d\tEmbed dim: %d\tLayers: %d\tTrain acc: %.2f\tVal acc: %.2f\tEpochs: %d\n' %(dataset, d, l, tr_acc, val_acc, epoch))
            f.close()
            print('Dataset %d\tEmbed dim: %d\tLayers: %d\tTrain acc: %.2f\tVal acc: %.2f\tEpochs: %d' %(dataset, d, l, tr_acc, val_acc, epoch))
            if val_acc > best_acc:
                best_acc = val_acc
                best_embed_size = d
                best_layers = l
    
    f = open(file_name, "a")
    f.write('\nBest Validation accuracy: %.2f\tEmbed dim: %d\tLayers: %d' %(best_acc, best_embed_size, best_layers))
    f.close()
    print('\nBest Validation accuracy: %.2f\tEmbed dim: %d\tLayers: %d' %(best_acc, best_embed_size, best_layers))


def random_search():
    # choose dataset
    dataset = 5

    # select best grid search values
    embed_size = 192
    layers = 2

    # define variables to perform random search
    n_runs = 20
    lr = np.logspace(start=-4, stop=-1.5, base=10, num=n_runs)  # log-spaced values between 1e-4 and 1e-(1.5)
    weight_decay = np.logspace(start=-7, stop=-3, base=10, num=n_runs)   
    dropout = np.random.uniform(low=0, high=0.5, size=n_runs) 

    # random permutation
    lr = lr[np.random.permutation(n_runs)]
    wd = weight_decay[np.random.permutation(n_runs)]
    dropout = dropout[np.random.permutation(n_runs)]

    train_accs = np.empty(n_runs)
    val_accs = np.empty(n_runs)
    epochs_training = np.empty(n_runs)

    best_acc = 0
    indx_best = 0

    file_name = "search_results/random_search_" + str(int(time.time())) + ".txt"
    f = open(file_name, "a")
    f.write('Random search results for\tDataset: %d\tEmbed dim: %d\tLayers: %d\n' % (dataset, embed_size, layers))
    f.close()

    for i in range(n_runs):
        tr_acc, val_acc, epoch = train(dataset, embed_size, layers, lr=lr[i], wd=wd[i], dropout=dropout[i])
        train_accs[i] = tr_acc
        val_accs[i] = val_acc
        epochs_training[i] = epoch
        f = open(file_name, "a")
        f.write('Learning rate: %.5f\tWeight decay: %.8f\tDropout: %.2f\tTrain acc: %.2f\tVal acc: %.2f\tEpochs: %d\n' %(lr[i], wd[i], dropout[i], tr_acc, val_acc, epoch))
        f.close()
        print('Learning rate: %.5f\tWeight decay: %.8f\tDropout: %.2f\tTrain acc: %.2f\tVal acc: %.2f\tEpochs: %d\n' %(lr[i], wd[i], dropout[i], tr_acc, val_acc, epoch))
        if val_acc > best_acc:
            best_acc = val_acc
            indx_best = i
    
    f = open(file_name, "a")
    f.write('\nBest Validation accuracy: %.2f\tLearning rate: %.5f\tWeight decay: %.8f\tDropout: %.2f' %(best_acc, lr[indx_best], wd[indx_best], dropout[indx_best]))
    f.close()
    print('\nBest Validation accuracy: %.2f\tLearning rate: %.5f\tWeight decay: %.8f\tDropout: %.2f' %(best_acc, lr[indx_best], wd[indx_best], dropout[indx_best]))


def train(dataset=8, hidden_dim=64, layers=2, max_epochs=200, lr=0.01, wd=1e-5, dropout=0.2, early_stopping=15, seed=42):
    np.random.seed(42)
    sys.argv = [sys.argv[0]]

    parser = ArgumentParser(description='Train a model')
    parser.add_argument('--dataset', type=int, default=dataset)
    parser.add_argument('--hidden-dim', type=int, default=hidden_dim)
    parser.add_argument('--lstm-layers', type=int, default=layers)
    parser.add_argument('--max-epochs', type=int, default=max_epochs)
    parser.add_argument('--early-stopping', type=int, default=early_stopping)
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--wd', type=float, default=wd)
    parser.add_argument('--dropout', type=float, default=dropout)
    parser.add_argument('--use-saved-dataset', type=bool, default=True)
    parser.add_argument('--save-path', type=str, default=None)
    parser.add_argument('--load-path', type=str, default=None)
    
    args = parser.parse_args()
    args.dataset = dataset
    args.hidden_dim = hidden_dim
    args.layers = layers
    return train_model(args)


if __name__ == "__main__":

    if sys.argv[1] == 'grid':
        grid_search()
    elif sys.argv[1] == 'random':
        random_search()
    else:
        print('Wrong argument. Either \'grid\' or \'random\'')
