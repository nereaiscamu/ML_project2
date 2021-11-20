from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np
from mock_dataset import get_mock_dataset
from lstm_chord_model import LSTMChordModel
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
def evaluate_model(model, test_loader):
    for batch_idx, batch in enumerate(test_loader):
        inputs = batch["input"].float()
        targets = batch["target"]
        lengths = batch["length"]

        preds = model(inputs, lengths)
        pdb.set_trace()
         

if __name__ == "__main__":
    # Get dataset
    train_dataset, test_dataset, vocab_size = get_mock_dataset()
    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True) 
    test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False) 

    # Create model
    model = LSTMChordModel(vocab_size, lstm_hidden_size=16)

    # Define training variables
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    epochs = 0

    # TRAIN
    for epoch in range(epochs):
        print("EPOCH", epoch)
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input"].float()
            targets = batch["target"]
            lengths = batch["length"]

            optimizer.zero_grad()
            outputs = model(inputs, lengths)

            print(inputs.shape)
            print(targets.shape)
            print(outputs.shape)

            loss = get_loss_vl(outputs, targets)
            print("Loss:", loss)
            acc = get_accuracy_vl(outputs, targets)
            print("Acc:", acc)

            loss.backward()
            optimizer.step() 


    # EVALUATE
    print('*** Training done! ***')
    evaluate_model(model, test_loader)