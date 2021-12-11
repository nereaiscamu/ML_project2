from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./Data/')
from one_hot_encoding import get_dataset_one_hot
from multi_hot_encoding import get_dataset_multi_hot
from models.lstm_chord_models import LSTMChord, LSTMChordEmbedding, LSTMChordEmbedding_Multihot
from models.lstm_melody_models import LSTM_Multihot, LSTM_Multihot_MLP
from argparse import ArgumentParser
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

def get_val_loss(model, val_dataset):
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False) 

    for batch_idx, batch in enumerate(val_loader):
        inputs = batch["input"].float()
        targets = batch["target"]
        lengths = batch["length"]

        outputs = model(inputs, lengths)
        loss = get_loss_vl(outputs, targets)
    
    return loss.item()


# DM
def evaluate_model(model, dataset):
    # TODO idk if this is the best way to load test data. what is the role of batch_size here?
    model.eval()
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False) 

    correct = 0
    total = 0
    for batch_idx, batch in enumerate(loader):
        inputs = batch["input"].float()
        targets = batch["target"]
        lengths = batch["length"]

        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        targets = targets.flatten()

        # Mask the outputs and targets
        mask = targets != -1
        
        correct += (preds[mask] == targets[mask]).sum()
        total += sum(mask)
    

    acc = 100 * correct/total    
    return acc
         

def train(args):
    # Get dataset
    # 354 training samples
    batch_size = 20

    #train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot_new_encoding(choice=2)
    train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

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
    epochs = 50
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # TRAIN
    n_batches = np.ceil(len(train_dataset)/batch_size)
    for epoch in range(epochs):
        model.train()
        print("EPOCH", epoch)
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
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
        train_losses.append(epoch_loss)
        val_losses.append(get_val_loss(model, val_dataset))
        train_accuracies.append(evaluate_model(model, train_dataset))
        val_accuracies.append(evaluate_model(model, val_dataset))
        print("Train/val loss: \t%.4f\t%.4f\t\tTrain/val accuracy: \t%.2f\t%.2f" % (epoch_loss, val_losses[-1], train_accuracies[-1], val_accuracies[-1]))

        if epoch == 50 or epoch == 100 or epoch == 150:
            for g in optimizer.param_groups:
                g['lr'] /= 2

    print('*** Training done! ***')

    # EVALUATE
    tr_acc = evaluate_model(model, train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    val_acc = evaluate_model(model, val_dataset)
    print('Val accuracy:\t%.2f' % val_acc)
    te_acc = evaluate_model(model, test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('loss.png')
    plt.show()

    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig('acc.png')
    plt.show()

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

def load_model(load_path):
    train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=4)

    # Create model
    #model = LSTMChord(vocab_size, lstm_hidden_size=16)
    #model = LSTMChordEmbedding(vocab_size, embed_size=16, lstm_hidden_size=16)
    model = LSTM_Multihot(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)
    #model = LSTM_Multihot_MLP(input_size, embed_size=64, lstm_hidden_size=64, target_size=target_size, num_layers=2, dropout_linear=0.4, dropout_lstm=0.4)

    model.load_state_dict(torch.load(load_path))

    # EVALUATE
    tr_acc = evaluate_model(model, train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    val_acc = evaluate_model(model, val_dataset)
    print('Val accuracy:\t%.2f' % val_acc)
    te_acc = evaluate_model(model, test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)




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
                        #default=None,
                        default='models/trained_models/model_name.pth',
                        help='')

    args = parser.parse_args()

    if args.load_path is not None:
        load_model(args.load_path)
    else:
        train(args)
