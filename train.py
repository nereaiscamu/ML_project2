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
        #train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot_new_encoding(choice=2)
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
                            dropout_linear=0.2, 
                            dropout_lstm=0.2)
    #model = LSTM_Multihot_MLP(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)
    model = model.to(device)

    # Define training variables
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    #optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    #optimizer = optim.SGD(model.parameters(), lr=0.0005)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

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
        print("EPOCH %d\tTrain/val loss: \t%.4f\t%.4f\t\tTrain/val accuracy: \t%.2f\t%.2f" % (epoch, epoch_loss, val_losses[-1], train_accuracies[-1], val_accuracies[-1]))

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

def load_model(load_path):
    train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=7)

    len_sequences = len(train_dataset) + len(val_dataset) + len(test_dataset)
    random_idxs = np.random.RandomState(seed=42).permutation(len_sequences)
    test_split = random_idxs[int(len_sequences*0.9):]

    # Create model
    #model = LSTMChord(vocab_size, lstm_hidden_size=16)
    #model = LSTMChordEmbedding(vocab_size, embed_size=16, lstm_hidden_size=16)
    model = LSTM_Multihot(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2)
    #model = LSTM_Multihot_MLP(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)

    model.load_state_dict(torch.load(load_path))

    # EVALUATE
    tr_acc = evaluate_model(model, train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    val_acc = evaluate_model(model, val_dataset)
    print('Val accuracy:\t%.2f' % val_acc)
    te_acc = evaluate_model(model, test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)

    with open('models/new_chord_map.pkl', 'rb') as f:
        new_chord_map = pickle.load(f)
        new_chord_map = dict((v,k) for k,v in new_chord_map.items())
    
    song_list = list()
    song_length = list()
    song_accuracy = list()
    preds_total = [[]]
    targets_total = [[]]

    for i, song in enumerate(test_dataset):
        inputs = song["input"].float().unsqueeze(0)   # need to add dim for batch_size=1
        targets = song["target"]
        lengths = [song["length"]]

        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        targets = targets.flatten()
        mask = targets != -1
        
        correct = (preds == targets[mask]).sum()
        correct = correct.float()
        acc = correct/sum(mask) * 100
        
        preds_chord = [new_chord_map[key.item()] for key in preds]
        targets_chord = [new_chord_map[key.item()] for key in targets[mask]]
        
        song_list.append(test_split[i]+1)
        song_length.append(int(lengths[0]))
        song_accuracy.append(round(float(acc),2))
        preds_total.append(pd.DataFrame(preds_chord))
        targets_total.append(pd.DataFrame(targets_chord))
        print('Test song %d\tSong ID: %d\tLength: %d\tAccuracy: %.2f' % (i, test_split[i]+1, lengths[0], acc))
    

    # QUALITATIVE STUDY
    while True:
        print('\nTest dataset of length %d. Enter the index of a sample or (q)uit:' % len(test_dataset))
        input_ = input()
        if input_ == 'q':
            break
        sample_id = int(input_)
        assert sample_id < len(test_dataset) and sample_id >= 0, 'Invalid sample index'

        sample = test_dataset.__getitem__(sample_id)
        inputs = sample["input"].float().unsqueeze(0)   # need to add dim for batch_size=1
        targets = sample["target"]
        lengths = [sample["length"]]

        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        preds_chord = [new_chord_map[key.item()] for key in preds]
        targets = targets.flatten()
        # Mask the outputs and targets
        mask = targets != -1
        targets_chord = [new_chord_map[key.item()] for key in targets[mask]]

        correct = (preds == targets[mask]).sum()
        correct = correct.float()
        acc = correct/sum(mask) * 100

        print('Number chords in the song: ', lengths[0])
        print('\nPredictions') 
        print(preds_chord)
        print('\nTargets') 
        print(targets_chord)
        print('\nAccuracy in this song: %.2f\n' % acc.item())
        
        return song_list, song_length, song_accuracy, preds_total, targets_total


if __name__ == "__main__":
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
                        default=False,
                        help='')
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

    if args.load_path is not None:
        load_model(args.load_path)
    else:
        train(args)