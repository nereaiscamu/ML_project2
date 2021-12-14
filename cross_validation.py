from torch.utils import data
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import collections
import matplotlib.pyplot as plt
import sys
sys.path.append('./Data/')
from one_hot_encoding import get_dataset_one_hot
from multi_hot_encoding import get_dataset_multi_hot, get_dataset_multi_hot_without_split
from models.lstm_chord_models import LSTMChord, LSTMChordEmbedding, LSTMChordEmbedding_Multihot
from models.lstm_melody_models import LSTM_Multihot, LSTM_Multihot_MLP
from argparse import ArgumentParser
from sklearn.model_selection import KFold
import pdb

# Andrew's
def get_loss_vl(outputs, targets):
  # Transpose because torch expects dim 1 to contain the classes
  # Add ignore_index
  return F.cross_entropy(outputs.transpose(1, 2), targets, ignore_index=-1)

# Andrew's
def get_accuracy_vl(outputs, targets):
  flat_outputs = outputs.argmax(dim=2).flatten()
  flat_targets = targets.flatten()

  # Mask the outputs and targets
  mask = flat_targets != -1

  return 100 * (flat_outputs[mask] == flat_targets[mask]).sum() / sum(mask)

def get_val_loss(model, val_loader):
    model.eval()
    #val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) 

    for batch_idx, batch in enumerate(val_loader):
        inputs = batch["input"].float()
        lengths = batch["length"]
        targets = batch["target"][:, :max(lengths)]

        outputs = model(inputs, lengths)
        loss = get_loss_vl(outputs, targets)
    
    return loss.item()


# DM
def evaluate_model(model, loader=None, dataset=None):
    # TODO idk if this is the best way to load test data. what is the role of batch_size here?
    model.eval()
    if dataset != None:
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) 

    assert loader != None

    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].float()
        lengths = batch["length"]
        targets = batch["target"][:, :max(lengths)]

        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        targets = targets.flatten()

        # Mask the outputs and targets
        mask = targets != -1
        
        correct += (preds[mask] == targets[mask]).sum()
        total += sum(mask)
    

    acc = 100 * correct.item()/total    
    return acc


def split_dataset(dataset, val_split=0.1, test_split=0.1):
    """
    Splits the dataset in 80% train, 10% validation, 10% test
    """
    # Split Train/Val/Test
    len_dataset = len(dataset)
    # random_idxs = np.random.RandomState(seed=42).permutation(len_dataset)    # this randomState has a localized effect, so the permutation will be the same always (and can use test set in load_model)
    # split_1 = int(len_dataset*(1-test_split-val_split))
    # split_2 = int(len_dataset*(1-test_split))

    # train_idxs = random_idxs[:split_1]
    # val_idxs = random_idxs[split_1:split_2]
    # test_idxs = random_idxs[split_2:]

    # train_set = dataset[train_idxs]
    # val_set = dataset[val_idxs]
    # test_set = dataset[test_idxs]

    len_tr = int(0.8*len_dataset)
    len_rem = len_dataset - len_tr
    train_set, rem_set = torch.utils.data.random_split(dataset, [len_tr, len_rem])

    len_rem_set = len(rem_set)
    len_val = int(0.5*len_rem_set)
    len_test = len_rem_set - len_val
    val_set, test_set = torch.utils.data.random_split(rem_set, [len_val, len_test])

    return train_set, val_set, test_set

def train(args):
    # Get dataset
    # 354 training samples
    # Configurations
    batch_size = 20
    k_folds = 4
    epochs = 100
    early_stopping = 15
    torch.manual_seed(42)

    #train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot_new_encoding(choice=2)
    #train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=8)
    dataset, input_size, target_size = get_dataset_multi_hot_without_split(choice=7)
    train_set, val_set, test_set = split_dataset(dataset)
    print('Dataset length:   %d' % len(dataset))
    print('Train set length: %d' % len(train_set))
    print('Val set length:   %d' % len(val_set))
    print('Test set length:  %d' % len(test_set))
    
    # Create set with the train_set and val_set for the cross validation
    cross_set = ConcatDataset([train_set, val_set])

    #kFold
    kfold = KFold(n_splits=k_folds, shuffle=True)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    models = []

    # TRAIN
    n_batches = np.ceil(len(cross_set)/batch_size)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, val_ids) in enumerate(kfold.split(cross_set)):
        print(f'\nFOLD {fold+1}')
        print('--------------------------------')
        train_losses_fold = []
        val_losses_fold = []
        train_accuracies_fold = []
        val_accuracies_fold = []
        losses = collections.deque(maxlen=early_stopping)

        # Sample elements randomly from a given list of ids, no replacement.
        #train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        #val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

        #train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler) 
        train_loader = DataLoader(cross_set, batch_size=batch_size, sampler=train_ids) 
        #val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_subsampler)
        val_loader = DataLoader(cross_set, batch_size=batch_size, sampler=val_ids)

        # Create model
        #model = LSTMChord(vocab_size, lstm_hidden_size=16)
        #model = LSTMChordEmbedding(vocab_size, embed_size=16, lstm_hidden_size=16)
        model = LSTM_Multihot(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)
        #model = LSTM_Multihot_MLP(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)

        # Define training variables
        #optimizer = optim.Adam(model.parameters(), lr=0.01)
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
        #optimizer = optim.SGD(model.parameters(), lr=0.0005)
        #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for _, batch in enumerate(train_loader, 0):
                inputs = batch["input"].float()
                lengths = batch["length"]
                targets = batch["target"][:, :max(lengths)] 

                optimizer.zero_grad()
                outputs = model(inputs, lengths)

                loss = get_loss_vl(outputs, targets)
    
                loss.backward()

                optimizer.step() 

                epoch_loss += loss.item() / n_batches

            model.eval()
            train_losses_fold.append(epoch_loss)
            val_losses_fold.append(get_val_loss(model, val_loader))
            train_accuracies_fold.append(evaluate_model(model, loader=train_loader))
            val_accuracies_fold.append(evaluate_model(model, loader=val_loader))
            print("EPOCH %.2d\tTrain/val loss: \t%.4f\t%.4f\t\tTrain/val accuracy: \t%.2f\t%.2f" % (epoch, epoch_loss, val_losses_fold[-1], train_accuracies_fold[-1], val_accuracies_fold[-1]))

            losses.append(val_losses_fold[-1])
            if len(losses) == early_stopping and losses[-1] >= max(losses):
                break

            if epoch == 50 or epoch == 100 or epoch == 150:
                for g in optimizer.param_groups:
                    g['lr'] /= 2
        
        models.append(model)
        
        # EVALUATE
        tr_acc = evaluate_model(model, dataset=train_set)
        print('\nTrain accuracy:\t%.2f' % tr_acc)
        val_acc = evaluate_model(model, dataset=val_set)
        print('Val accuracy:\t%.2f' % val_acc)
        te_acc = evaluate_model(model, dataset=test_set)
        print('Test accuracy:\t%.2f\n' % te_acc)

        train_losses.append(train_losses_fold)
        val_losses.append(val_losses_fold)
        train_accuracies.append(train_accuracies_fold)     
        val_accuracies.append(val_accuracies_fold)     
        test_accuracies.append(te_acc)    

        plot_loss(train_losses_fold, val_losses_fold, name=(f'loss_{fold+1}.png'))
        plot_accuracy(train_accuracies_fold, val_accuracies_fold, name=(f'acc_{fold+1}.png')) 

    print('\n*** Training done for all %d folds! ***\n' % k_folds)

    print("Average Train/val loss: \t%.4f\t%.4f\t\tAverage Train/val accuracy: \t%.2f\t%.2f" % (np.mean(train_losses), np.mean(val_losses), np.mean(train_accuracies), np.mean(val_accuracies)))
    print("Average test accuracy:  \t%.2f\n" % np.mean(test_accuracies))

    plot_losses(train_losses, val_losses)
    plot_accuracies(train_accuracies, val_accuracies)

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)



def plot_loss(train_losses, val_losses, name='loss.png'):
    """
    Plots the losses and saves the plot to 'figs_results/loss.png'
    """
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('figs_results/' + name)
    plt.show()

def plot_accuracy(train_accuracies, val_accuracies, name='acc.png'):
    """
    Plots the accuracies and saves the plot to 'figs_results/acc.png'
    """
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validaton')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figs_results/' + name)
    plt.show()


def plot_losses(train_losses, val_losses, name='loss.png'):
    """
    Plots the losses and saves the plot to 'figs_results/loss.png'
    """
    for i in range(len(train_losses)):
        plt.plot(train_losses[i], label='Train_%d' % i)

    for i in range(len(val_losses)):
        plt.plot(val_losses[i], label='Val_%d' % i)

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('figs_results/' + name)
    plt.show()

def plot_accuracies(train_accuracies, val_accuracies, name='acc.png'):
    """
    Plots the accuracies and saves the plot to 'figs_results/acc.png'
    """
    for i in range(len(train_accuracies)):
        plt.plot(train_accuracies[i], label='Train_%d' % i)

    for i in range(len(val_accuracies)):
        plt.plot(val_accuracies[i], label='Val_%d' % i)

    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figs_results/' + name)
    plt.show()

if __name__ == "__main__":
    np.random.seed(42)

    parser = ArgumentParser(description='Train a model')
    parser.add_argument('--save-path', type=str,
                        # required=True,
                        #default=None,
                        default='models/trained_models/model_name.pth',
                        help='')
    parser.add_argument('--load-path', type=str,
                        # required=True,
                        default=None,
                        #default='models/trained_models/model_1_dataset_1_s42.pth',
                        help='')

    args = parser.parse_args()

    # if args.load_path is not None:
    #     load_model(args.load_path)
    # else:
    train(args)
