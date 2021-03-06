import torch
import numpy as np
import sys
sys.path.append('./data/')
from multi_hot_encoding import get_dataset_multi_hot
from models.lstm_melody_models import LSTM_Multihot
from helpers import evaluate_model
import pickle
import pandas as pd

import pdb


def load_model(load_path, dataset, hidden_dim, layers, seed=42, song_input=True):

    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    device = torch.device("cpu")
    
    train_dataset, val_dataset, test_dataset, input_size, target_size, tune_ids = get_dataset_multi_hot(choice=dataset, seed=seed, get_tune_ids=True)
    
    len_sequences = len(train_dataset) + len(val_dataset) + len(test_dataset)
    random_idxs = np.random.RandomState(seed=seed).permutation(len_sequences)
    test_split = random_idxs[int(len_sequences*0.9):]

    # Load model
    model = LSTM_Multihot(input_size, embed_size=hidden_dim, lstm_hidden_size=hidden_dim, target_size=target_size, num_layers=layers)
    model.load_state_dict(torch.load(load_path))

    # evaluate
    tr_acc = evaluate_model(model, device, dataset=train_dataset)
    print('Train accuracy:\t%.2f' % tr_acc)
    val_acc = evaluate_model(model, device, dataset=val_dataset)
    print('Val accuracy:\t%.2f' % val_acc)
    te_acc = evaluate_model(model, device, dataset=test_dataset)
    print('Test accuracy:\t%.2f' % te_acc)

    # Load chord map -- from one-hot to chord name
    with open('Chord_Vocab/chord_map.pkl', 'rb') as f:
        chord_map = pickle.load(f)
        chord_map = dict((v,k) for k,v in chord_map.items())
    
    tune_list = list()
    tune_length = list()
    tune_accuracy = list()
    preds_total = [[]]
    targets_total = [[]]

    # Print accuracy song-wise
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
        
        preds_chord = [chord_map[key.item()] for key in preds]
        targets_chord = [chord_map[key.item()] for key in targets[mask]]
        
        tune_list.append(tune_ids[test_split[i]])
        tune_length.append(int(lengths[0]))
        tune_accuracy.append(round(float(acc),2))
        preds_total.append(pd.DataFrame(preds_chord))
        targets_total.append(pd.DataFrame(targets_chord))
        print('Test song %d\tSong ID: %d\tLength: %d\tAccuracy: %.2f' % (i, tune_ids[test_split[i]], lengths[0], acc))
    

    # Qualitative study of ONE SONG
    while True and song_input:
        print('\nTest dataset of length %d. Enter the index of a sample, or \'enter\' to skip :' % len(test_dataset))
        input_ = input()
        if input_ == '':
            print("Skipped current dataset, continuing now!")
            break
        sample_id = int(input_)
        assert sample_id < len(test_dataset) and sample_id >= 0, 'Invalid sample index'

        sample = test_dataset.__getitem__(sample_id)
        inputs = sample["input"].float().unsqueeze(0)   # need to add dim for batch_size=1
        targets = sample["target"]
        lengths = [sample["length"]]
        preds = model(inputs, lengths)
        preds = preds.argmax(dim=2).flatten()
        preds_chord = [chord_map[key.item()] for key in preds]
        targets = targets.flatten()
        mask = targets != -1                            # Mask the outputs and targets
        targets_chord = [chord_map[key.item()] for key in targets[mask]]

        correct = (preds == targets[mask]).sum()
        correct = correct.float()
        acc = correct/sum(mask) * 100

        print('Number chords in the song: ', lengths[0])
        print('\nPredictions') 
        print(preds_chord)
        print('\nTargets') 
        print(targets_chord)
        print('\nAccuracy in this song: %.2f\n' % acc.item())
    
    return tune_list, tune_length, tune_accuracy, preds_total, targets_total


def load_training_data(dataset, seed=42):
    train_dataset, _, _, _, _, _= get_dataset_multi_hot(choice=dataset, seed=seed, get_tune_ids=True)
    
    # Load chord map -- from one-hot to chord name
    with open('Chord_Vocab/chord_map.pkl', 'rb') as f:
        chord_map = pickle.load(f)
        chord_map = dict((v,k) for k,v in chord_map.items())
    
    song_length = list()
    targets_total = [[]]

    # Print accuracy song-wise
    for i, song in enumerate(train_dataset):
        targets = song["target"]
        lengths = [song["length"]]

        targets = targets.flatten()
        mask = targets != -1
                
        targets_chord = [chord_map[key.item()] for key in targets[mask]]
        
        song_length.append(int(lengths[0]))
        targets_total.append(pd.DataFrame(targets_chord))
        
    
    return song_length, targets_total
