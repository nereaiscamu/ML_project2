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
    batch_size = 5
    avg_loss_steps = 20

    # train_dataset, test_dataset, vocab_size = get_mock_dataset()
    #train_dataset, test_dataset, vocab_size = get_dataset(choice=1, test_split=0.2)
    train_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=6, test_split=0.2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    # Create model
    #model = LSTMChord(vocab_size, lstm_hidden_size=16)
    #model = LSTMChordEmbedding(vocab_size, embed_size=16, lstm_hidden_size=16)
    model = LSTMChordEmbedding_Multihot(input_size, embed_size=16, lstm_hidden_size=16, target_size=target_size)

    # Define training variables
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 20
    losses = []
    train_accuracies = []
    test_accuracies = []

    # TRAIN
    for epoch in range(epochs):
        print("EPOCH", epoch)
        epoch_loss = 0
        avg_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].float()
            lengths = batch["length"]
            targets = batch["target"][:, :max(lengths)] 

            optimizer.zero_grad()
            outputs = model(inputs, lengths)

            loss = get_loss_vl(outputs, targets)
   
            loss.backward()
            optimizer.step() 

            epoch_loss += loss.item()/len(train_dataset)
            avg_loss += loss.item()
            #print("Loss:", loss.item())
            if (batch_idx+1) % avg_loss_steps == 0:
                losses.append(avg_loss / avg_loss_steps)
                avg_loss = 0

        print("Epoch avg loss: %.4f" % epoch_loss)
        train_accuracies.append(evaluate_model(model, train_dataset))
        test_accuracies.append(evaluate_model(model, test_dataset))

    print('*** Training done! ***')

    # EVALUATE
    tr_acc = evaluate_model(model, train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    te_acc = evaluate_model(model, test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)

    plt.plot(losses)
    plt.ylabel('Training loss')
    plt.show()
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
