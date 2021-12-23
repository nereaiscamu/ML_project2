from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('./data/')
from multi_hot_encoding import get_dataset_multi_hot
from models.lstm_melody_models import LSTM_Multihot
from argparse import ArgumentParser
from helpers import *
import pickle
import pandas as pd
# Suppress warning "A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None  # default='warn'


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
        for _, batch in enumerate(train_loader):
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
        
        val_losses.append(get_val_loss(model, device, val_set=val_dataset))
        train_accuracies.append(evaluate_model(model, device, dataset=train_dataset))
        val_accuracies.append(evaluate_model(model, device, dataset=val_dataset))

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
        
        # if not learning, stop (for random search)
        if epoch > 40 and val_accuracies[-1] < 10:
            break

        print("EPOCH %.2d\tTrain/val loss: %.2f / %.2f\tLower loss: %.2f\tTrain/val accuracy: \t%.2f / %.2f" % (epoch, epoch_loss, val_losses[-1], best_cost, train_accuracies[-1], val_accuracies[-1]))

        
        if epoch == 50 or epoch == 100 or epoch == 150:
            for g in optimizer.param_groups:
                g['lr'] /= 2
        

    print('*** Training done! ***')

    # EVALUATE
    tr_acc = evaluate_model(model, device, dataset=train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    val_acc = evaluate_model(model, device, dataset=val_dataset)
    print('Val accuracy:\t%.2f' % val_acc)
    te_acc = evaluate_model(model, device, dataset=test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)

    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    if args.save_path is not None:
        torch.save(model.state_dict(), args.save_path)

    return tr_acc, val_acc, epoch

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