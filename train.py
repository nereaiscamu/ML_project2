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
import pandas as pd

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

def get_val_loss(model, val_dataset, device):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) 

    for batch_idx, batch in enumerate(val_loader):
        inputs = batch["input"].float().to(device)
        targets = batch["target"].to(device)
        lengths = batch["length"]

        outputs = model(inputs, lengths)
        loss = get_loss_vl(outputs, targets)
    
    return loss.item()


# DM
def evaluate_model(model, dataset, device):
    # TODO idk if this is the best way to load test data. what is the role of batch_size here?
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) 

    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].float().to(device)
        targets = batch["target"].to(device)
        lengths = batch["length"]

        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        targets = targets.flatten()

        # Mask the outputs and targets
        mask = targets != -1
        
        correct += (preds[mask] == targets[mask]).sum()
        total += sum(mask)
    

    acc = 100 * correct.item()/total    
    return acc
         

def train(args):
    # Get dataset
    # 354 training samples
    batch_size = 20

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    if args.use_saved_dataset:
        with open('data/datasets/dataset' + str(args.dataset) + '.pickle', 'rb') as f:
            (train_dataset, val_dataset, test_dataset, input_size, target_size) = pickle.load(f)
        print('*** Dataset ' + str(args.dataset) + ' loaded from file ***')
    else:
        train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=args.dataset, seed=args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    # Create model
    #model = LSTMChord(vocab_size, lstm_hidden_size=16)
    #model = LSTMChordEmbedding(vocab_size, embed_size=16, lstm_hidden_size=16)
    model = LSTM_Multihot(input_size, 
                            embed_size=args.hidden_dim, 
                            lstm_hidden_size=args.hidden_dim, 
                            target_size=target_size, 
                            num_layers=args.lstm_layers, 

                            dropout_linear=args.dropout, 
                            dropout_lstm=args.dropout)

    #model = LSTM_Multihot_MLP(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)
    model = model.to(device)

    # Define training variables
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    epochs = args.max_epochs
    best_cost = 1000
    early_stopping = args.early_stopping
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # TRAIN
    n_batches = np.ceil(len(train_dataset)/batch_size)
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
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
        train_losses.append(epoch_loss)
        
        val_losses.append(get_val_loss(model, val_dataset, device))
        train_accuracies.append(evaluate_model(model, train_dataset, device))
        val_accuracies.append(evaluate_model(model, val_dataset, device))

        # Early stopping based on the validation set:
        # Check that improvement has been made in the last X epochs
        if val_losses[-1] < best_cost:
            best_cost = val_losses[-1]
            last_improvement = 0
        else:
            last_improvement +=1
            if last_improvement > early_stopping:
                print("\nNo improvement found during the last %d epochs, stopping optimization.\n" % early_stopping)
                break
        
        print("EPOCH %d\tTrain/val loss: %.2f / %.2f\tLower loss: %.2f\tTrain/val accuracy: \t%.2f / %.2f" % (epoch, epoch_loss, val_losses[-1], best_cost, train_accuracies[-1], val_accuracies[-1]))

        
        if epoch == 50 or epoch == 100 or epoch == 150:
            for g in optimizer.param_groups:
                g['lr'] /= 2
        

    print('*** Training done! ***')

    # EVALUATE
    tr_acc = evaluate_model(model, train_dataset, device)
    print('Train accuracy:\t%.2f' % tr_acc)
    val_acc = evaluate_model(model, val_dataset, device)
    print('Val accuracy:\t%.2f' % val_acc)
    te_acc = evaluate_model(model, test_dataset, device)
    print('Test accuracy:\t%.2f' % te_acc)

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('figs_results/loss.png')
    plt.show()

    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('figs_results/acc.png')
    plt.show()

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

    return tr_acc, val_acc, epoch

if __name__ == "__main__":
    np.random.seed(42)

    parser = ArgumentParser(description='Train a model')

    parser.add_argument('--dataset', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=192)
    parser.add_argument('--lstm-layers', type=int, default=2)
    parser.add_argument('--max-epochs', type=int, default=200)
    parser.add_argument('--early-stopping', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lr', type=float, default=0.017)
    parser.add_argument('--wd', type=float, default=8e-5)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--use-saved-dataset', type=bool, default=False)


    parser.add_argument('--save-path', type=str,
                        #default=None,
                        default='models/trained_models/optimized_192_2_dataset_4.pth',
                        help='')

    args = parser.parse_args()

    train(args)