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
from models.lstm_melody_models import LSTM_Multihot
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

def get_test_loss(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False) 

    for batch_idx, batch in enumerate(test_loader):
        inputs = batch["input"].float()
        targets = batch["target"]
        lengths = batch["length"]

        outputs = model(inputs, lengths)
        loss = get_loss_vl(outputs, targets)
    
    return loss.item()


# DM
def evaluate_model(model, test_dataset):
    # TODO idk if this is the best way to load test data. what is the role of batch_size here?
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False) 

    correct = 0
    total = 0
    for batch_idx, batch in enumerate(test_loader):
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
         

if __name__ == "__main__":
    # Get dataset
    # 354 training samples
    batch_size = 20

    # train_dataset, test_dataset, vocab_size = get_mock_dataset()
    #train_dataset, test_dataset, vocab_size = get_dataset(choice=1, test_split=0.2)
    #train_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot_new_encoding(choice=2, test_split=0.2)
    train_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=7, test_split=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    # Create model
    #model = LSTMChord(vocab_size, lstm_hidden_size=16)
    #model = LSTMChordEmbedding(vocab_size, embed_size=16, lstm_hidden_size=16)
    model = LSTM_Multihot(input_size, embed_size=128, lstm_hidden_size=128, target_size=target_size, num_layers=2)

    # Define training variables
    #optimizer = optim.Adam(model.parameters(), lr=0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.00001)
    #optimizer = optim.SGD(model.parameters(), lr=0.0005)
    #optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    epochs = 100
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # TRAIN
    n_batches = np.ceil(len(train_dataset)/batch_size)
    for epoch in range(epochs):
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

        train_losses.append(epoch_loss)
        test_losses.append(get_test_loss(model, test_dataset))
        train_accuracies.append(evaluate_model(model, train_dataset))
        test_accuracies.append(evaluate_model(model, test_dataset))
        print("Train/test loss: \t%.4f\t%.4f\t\tTrain/test accuracy: \t%.2f\t%.2f" % (epoch_loss, test_losses[-1], train_accuracies[-1], test_accuracies[-1]))

        #if epoch == 20 or epoch == 40:
        #    for g in optimizer.param_groups:
        #        g['lr'] /= 0.2

    print('*** Training done! ***')

    # EVALUATE
    tr_acc = evaluate_model(model, train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    te_acc = evaluate_model(model, test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)

    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()