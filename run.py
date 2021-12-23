import torch
import pickle
import sys

from torch.utils import data
sys.path.append('./data/')
from multi_hot_encoding import get_dataset_multi_hot
from models.lstm_melody_models import LSTM_Multihot
from helpers import evaluate_model


model_path = 'models/trained_models/1_Baseline.pth'
dataset = 1

model_path_mel = 'models/trained_models/2_Melody.pth'
dataset_mel = 2

model_path_mel_w = 'models/trained_models/6_Melody_weighted.pth'
dataset_mel_w = 6

model_path_mel_dur = 'models/trained_models/5_Melody_Duration.pth'
dataset_mel_dur = 5

model_path_bass = 'models/trained_models/3_Bass.pth'
dataset_bass = 3

model_path_mel_bass = 'models/trained_models/4_Melody_Bass.pth'
dataset_mel_bass = 4

use_saved_set=True


hidden_dim = 192
layers = 2
seed = 42

def compute_test_acc(model_path, dataset, use_saved_set=True):
    device = torch.device("cpu")
    
    if use_saved_set:
        with open('data/datasets/dataset%d.pickle' % dataset, 'rb') as f:
            (train_dataset, val_dataset, test_dataset, input_size, target_size) = pickle.load(f)
        print('*** Dataset loaded from file ***')
    else:
        train_dataset, val_dataset, test_dataset, input_size, target_size = get_dataset_multi_hot(choice=dataset, seed=seed)
    
    # Load model
    model = LSTM_Multihot(input_size, embed_size=hidden_dim, lstm_hidden_size=hidden_dim, target_size=target_size, num_layers=layers)
    model.load_state_dict(torch.load(model_path))

    # evaluate
    tr_acc = evaluate_model(model, device, dataset=train_dataset)
    print('Train accuracy:\t%.2f %%' % tr_acc)
    val_acc = evaluate_model(model, device, dataset=val_dataset)
    print('Val accuracy:\t%.2f %%' % val_acc)
    te_acc = evaluate_model(model, device, dataset=test_dataset)
    print('Test accuracy:\t%.2f %%' % te_acc)
    return te_acc
    
print('\nLoading Baseline:')
acc_baseline = compute_test_acc(model_path, dataset, use_saved_set=use_saved_set)
#print('Baseline Loaded\n')

print('\n\nLoading Melody:')
acc_mel = compute_test_acc(model_path_mel, dataset_mel, use_saved_set=use_saved_set)
#print('Melody Loaded\n')

print('\n\nLoading Melody weighted:')
acc_mel_w = compute_test_acc(model_path_mel_w, dataset_mel_w, use_saved_set=use_saved_set)
#print('Melody weighted Loaded\n')

print('\n\nLoading Melody + Duration:')
acc_mel_dur = compute_test_acc(model_path_mel_dur, dataset_mel_dur, use_saved_set=use_saved_set)
#print('Melody + Duration Loaded\n')

print('\n\nLoading Bass:')
acc_bass = compute_test_acc(model_path_bass, dataset_bass, use_saved_set=use_saved_set)
#print('Bass Loaded\n')

print('\n\nLoading Melody + Bass:')
acc_mel_bass = compute_test_acc(model_path_mel_bass, dataset_mel_bass, use_saved_set=use_saved_set)
#print('Melody + Bass: Loaded\n')



print('\n\n-----------------------------------------')
print('| Model\t\t\t| Test accuracy |')
print('-----------------------------------------')
print('| Baseline\t\t|    %.2f %% \t|' % acc_baseline)
print('-----------------------------------------')
print('| Melody\t\t|    %.2f %% \t|' % acc_mel)
print('| Melody weighted\t|    %.2f %% \t|' % acc_mel_w)
print('| Melody duration\t|    %.2f %% \t|' % acc_mel_dur)
print('| Bass\t\t\t|    %.2f %% \t|' % acc_bass)
print('| Melody + Bass\t\t|    %.2f %% \t|' % acc_mel_bass)
print('-----------------------------------------\n')
