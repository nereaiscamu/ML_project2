from torch.utils import data
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Data/')
from multi_hot_encoding import get_dataset_multi_hot
from models.lstm_melody_models import LSTM_Multihot
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from helpers import *
import pickle


def train(args):
    # Get dataset
    # 354 training samples
    # Configurations
    batch_size = 20
    k_folds = 4
    epochs = args.max_epochs
    early_stopping = args.early_stopping
    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.use_saved_dataset:
        with open('data/datasets/dataset_wo_split' + str(args.dataset) + '.pickle', 'rb') as f:
            dataset, input_size, target_size = pickle.load(f)
        print('*** Dataset without split ' + str(args.dataset) + ' loaded from file ***')
    else:
        dataset, input_size, target_size = get_dataset_multi_hot(choice=args.dataset, kfold=True)
    
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
        best_cost = 1000

        train_loader = DataLoader(cross_set, batch_size=batch_size, sampler=train_ids) 
        val_loader = DataLoader(cross_set, batch_size=batch_size, sampler=val_ids)

        # Create model
        model = LSTM_Multihot(input_size, 
                                    embed_size=args.hidden_dim, 
                                    lstm_hidden_size=args.hidden_dim, 
                                    target_size=target_size, 
                                    num_layers=args.lstm_layers, 

                                    dropout_linear=args.dropout, 
                                    dropout_lstm=args.dropout)

        model = model.to(device)

        # Define training variables
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)


        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for _, batch in enumerate(train_loader, 0):
                inputs = batch["input"].float().to(device)
                lengths = batch["length"]
                targets = batch["target"][:, :max(lengths)].to(device)

                optimizer.zero_grad()
                outputs = model(inputs, lengths)

                loss = get_loss_vl(outputs, targets)
    
                loss.backward()

                optimizer.step() 

                epoch_loss += loss.item() / n_batches

            model.eval()
            train_losses_fold.append(epoch_loss)
            val_losses_fold.append(get_val_loss(model, device, val_loader=val_loader))
            train_accuracies_fold.append(evaluate_model(model, device, loader=train_loader))
            val_accuracies_fold.append(evaluate_model(model, device, loader=val_loader))
            print("EPOCH %.2d\tTrain/val loss: \t%.4f\t%.4f\t\tTrain/val accuracy: \t%.2f\t%.2f" % (epoch, epoch_loss, val_losses_fold[-1], train_accuracies_fold[-1], val_accuracies_fold[-1]))

            # Early stopping based on the validation set:
            # Check that improvement has been made in the last X epochs
            if val_losses_fold[-1] < best_cost:
                best_cost = val_losses_fold[-1]
                last_improvement = 0
            else:
                last_improvement +=1
            if last_improvement > early_stopping:
                print("\nNo improvement found during the last %d epochs, stopping optimization.\n" % early_stopping)
                break

            if epoch == 50 or epoch == 100 or epoch == 150:
                for g in optimizer.param_groups:
                    g['lr'] /= 2
        
        models.append(model)
        
        # EVALUATE
        tr_acc = evaluate_model(model, device, dataset=train_set)
        print('\nTrain accuracy:\t%.2f' % tr_acc)
        val_acc = evaluate_model(model, device, dataset=val_set)
        print('Val accuracy:\t%.2f' % val_acc)
        te_acc = evaluate_model(model, device, dataset=test_set)
        print('Test accuracy:\t%.2f\n' % te_acc)

        train_losses.append(train_losses_fold)
        val_losses.append(val_losses_fold)
        train_accuracies.append(train_accuracies_fold)     
        val_accuracies.append(val_accuracies_fold)     
        test_accuracies.append(te_acc)    

        plot_loss(train_losses_fold, val_losses_fold, name=(f'loss_{fold+1}.png'))
        plot_accuracy(train_accuracies_fold, val_accuracies_fold, name=(f'acc_{fold+1}.png')) 

    print('\n*** Training done for all %d folds! ***\n' % k_folds)

    train_losses_mean = np.mean([sublist[-1] for sublist in train_losses]) 
    val_losses_mean = np.mean([sublist[-1] for sublist in val_losses]) 
    train_accuracies_mean = np.mean([sublist[-1] for sublist in train_accuracies]) 
    val_accuracies_mean = np.mean([sublist[-1] for sublist in val_accuracies]) 
    print("Average Train/val loss: \t%.4f\t%.4f\t\tAverage Train/val accuracy: \t%.2f\t%.2f" % (train_losses_mean, val_losses_mean, train_accuracies_mean, val_accuracies_mean))
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

    parser.add_argument('--dataset', type=int, default=1)
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--early-stopping', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.007)
    parser.add_argument('--wd', type=float, default=5e-6)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use-saved-dataset', type=bool, default=False)


    parser.add_argument('--save-path', type=str,
                        #default=None,
                        default='models/trained_models/optimized_192_2_dataset_1_new.pth',
                        help='')

    args = parser.parse_args()

    train(args)