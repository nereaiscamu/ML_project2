from argparse import ArgumentParser

from torch._C import device
from load_model import load_model
from load_model import load_training_data
import numpy as np
import plotly.express as px
import plotly.io as pio
import pandas as pd
# Suppress "A value is trying to be set on a copy of a slice from a DataFrame"
pd.options.mode.chained_assignment = None  # default='warn'
import pdb 
import os
import matplotlib.pyplot as plt
import seaborn as sn
from multi_hot_encoding import get_dataset_multi_hot
import pathlib 

# Select model. !! Use according dataset, hidden_dim, layers and seed !!

model_path = 'models/trained_models/1_Baseline.pth'
model_name = 'result_analysis/1_Baseline'
dataset = 1

model_path_mel = 'models/trained_models/2_Melody.pth'
model_name_mel = 'result_analysis/2_melody'
dataset_mel = 2

project_path = os.getcwd()
result_analysis_path = pathlib.os.path.join(project_path,'result_analysis/Report')

hidden_dim = 192
layers = 2
seed = 42


# Whether or not to generate the root pitch confusion matrix for all individial songs
do_conf_matrix_all_songs = False

# Whether or not to compare song accuracies
do_compare_accuracies = False

# general functions
correct_chords = lambda x: sum(x)/len(x)*100
num_chords = lambda x: len(x)
#%%

# Load models
song_list, song_length, song_accuracy, preds, targets = load_model(model_path, dataset, hidden_dim, layers, seed, song_input=False)
song_list_mel, song_length_mel, song_accuracy_mel, preds_mel, targets_mel = load_model(model_path_mel, dataset_mel, hidden_dim, layers, seed, song_input=False)
        

#%%

def compare_accuracies_table(song_list, song_length, song_accuracy,song_list_mel, song_length_mel, song_accuracy_mel):
    # Load baseline model
    song_df = pd.DataFrame()
    song_df['Test_sample_ID'] = range(len(targets)-1)
    song_df['Song'] = song_list
    song_df['Song_Length'] = song_length
    song_df['Song_Accuracy'] = song_accuracy
    # Load melody model
    song_mel_df = pd.DataFrame()
    song_mel_df['Test_sample_ID'] = range(len(targets_mel)-1)  
    song_mel_df['Song'] = song_list_mel 
    song_mel_df['Song_Length'] = song_length_mel
    song_mel_df['Song_Accuracy_mel'] = song_accuracy_mel
    song_df = song_df.merge(song_mel_df['Song_Accuracy_mel'], right_index=True, left_index=True)
    song_df['acc_diff'] = round((song_df['Song_Accuracy_mel'] - song_df['Song_Accuracy']),2)
    return(song_df)


def create_result_table(song_list, song_length, song_accuracy, preds, targets):
    results_numbers = pd.DataFrame()
    results_numbers['Song'] = song_list
    results_numbers['Song_Length'] = song_length
    results_numbers['Song_Accuracy'] = song_accuracy
    results_numbers['Test_sample_ID'] = range(len(targets)-1)
    
    result_table = pd.DataFrame(columns=['Test_sample_ID', 'Target_Chords', 'Pred_Chords'])
    for i, target in enumerate (targets):
        result_song = pd.DataFrame()
        result_song['Test_sample_ID'] = np.repeat(i-1, len(target))  # i-1 necessary because for some reason targets is shifted from [0,43] to [1,44]. targets[0] is empty
        result_song['Chord_idx'] = range(1, len(target)+1)
        result_song['Target_Chords'] = target
        result_song['Pred_Chords'] = preds[i]
        prev_chord = ['None']
        for z in range(1, len(target)):
            prev_chord.append(result_song['Target_Chords'][z-1])
        result_song['Previous_target'] = prev_chord
        result_table = pd.concat([result_table, result_song], axis=0)

    result_table = result_table.merge(results_numbers, left_on='Test_sample_ID', right_on='Test_sample_ID')
    result_table.loc[result_table['Target_Chords'] == result_table['Pred_Chords'], 'Correct'] = 1
    result_table.loc[result_table['Target_Chords'] != result_table['Pred_Chords'], 'Correct'] = 0
    result_table['Target_Seq'] =  result_table['Previous_target'].str.cat(result_table['Target_Chords'], sep = ' - ')

    # compute accuracy when previous chord was correct
    result_table['correct_prev_c'] = result_table.Correct.shift(1) * result_table.Correct
    result_table.loc[0, 'prec_correct'] = 0
    result_table['correct_prev_inc'] = - (result_table.Correct.shift(1)-1) * (result_table.Correct)
    result_table.loc[0, 'correct_prev_inc'] = 0

    return result_table, results_numbers

def create_train_table(dataset, seed):
    song_length_train, targets_train= load_training_data(dataset, seed=42)
    train_table = pd.DataFrame()
    train_table = pd.DataFrame(columns=['Train_sample_ID', 'Chord_idx', 'Target_Chords_Train'])
    
    for i, j  in enumerate (targets_train):
        input_song = pd.DataFrame()
        input_song['Train_sample_ID'] = np.repeat(i, len(j))
        input_song['Chord_idx'] = range(1, len(j)+1)
        input_song['Target_Chords_Train'] = j
        Prev_chord  = ['None']
        for z in range(1, len(j)):
            Prev_chord.append(input_song['Target_Chords_Train'][z-1])
        input_song['Prev_chord'] = Prev_chord
        train_table = pd.concat([train_table, input_song], axis=0)
        
    train_table['Target_Seq_train'] =  train_table['Prev_chord'].str.cat(train_table['Target_Chords_Train'],  sep = ' - ')
    return train_table
        



def train_sample_size_chords(train_table, size_seq = False):    
    train_chord_size = train_table
    if size_seq == False:
        train_chord_size['Target_Chords_Train2'] = train_chord_size['Target_Chords_Train']      
        train_chord_size = train_chord_size.groupby(
            by=["Target_Chords_Train"]).agg({'Target_Chords_Train2': num_chords }).reset_index()
        train_chord_size = train_chord_size.rename(columns={ 'Target_Chords_Train2': "Sample_Size_Train"})
    if size_seq == True:
        train_chord_size['Target_Seq_train2'] = train_chord_size['Target_Seq_train']      
        train_chord_size = train_chord_size.groupby(
            by=["Target_Seq_train"]).agg({'Target_Seq_train2': num_chords }).reset_index()
        train_chord_size = train_chord_size.rename(columns={ 'Target_Seq_train2': "Sample_Size_Train_Seq"})
            
        
    return train_chord_size

   

#%%

def F_score(result_table, train_chord_size):
    '''
    Precision is the number of true positive results divided by the number of all positive results,
    including those not identified correctly, and the recall is the number of true positive results divided 
    by the number of all samples that should have been identified as positive
      
     Precision is also known as positive predictive value,
      and recall is also known as sensitivity in diagnostic binary classification.
      
    # For prediction --> use the preds (corrects/total predicted)
    # For recall use the targets
    '''
    
    # 1- Compute recall 
    Recall_Chords = result_table.groupby(
        by=["Target_Chords"]).agg({'Correct': [num_chords, correct_chords] }).reset_index()
    Recall_Chords.columns = ["_".join(x) for x in Recall_Chords.columns.values]
    Recall_Chords = Recall_Chords.rename(columns={'Target_Chords_': 'Target_Chords', 
                                                  'Correct_<lambda_0>': "Sample_Size", 
                                                  'Correct_<lambda_1>': "recall"})
    # 2- Compute prediction 
    Prediction_Chords = result_table.groupby(
        by=["Pred_Chords"]).agg({'Correct': [num_chords, correct_chords] }).reset_index()
    Prediction_Chords.columns = ["_".join(x) for x in Prediction_Chords.columns.values]
    Prediction_Chords = Prediction_Chords.rename(columns={'Pred_Chords_': 'Pred_Chords', 
                                                          'Correct_<lambda_0>': "Sample_Size", 
                                                          'Correct_<lambda_1>': "precision"})
    # 3- Merge the tables to get the F-score
    # F_score = 2*((precision*recall)/(precision+recall))
    F_score = Recall_Chords.merge(Prediction_Chords, left_on='Target_Chords', right_on='Pred_Chords')
    F_score = F_score.assign(f_score = 2*((F_score['recall'] * F_score['precision'])/(F_score['recall'] + F_score['precision'])))
    F_score['f_score'] = F_score['f_score'].replace(np.nan, 0)
    F_score = F_score.sort_values(by = 'f_score')
    # 4- Merge with train data to know how many times each chord appeared in the training dataset
    F_score = F_score.merge(train_chord_size, left_on = 'Target_Chords', right_on = 'Target_Chords_Train' )
    return F_score


#%%
def accuracy_chord_idx(result_table):
    Acc_chord_idx = result_table.groupby(
        by=["Chord_idx"]).agg({'Correct': [num_chords, correct_chords] }).reset_index()
    Acc_chord_idx.columns = ["_".join(x) for x in Acc_chord_idx.columns.values]
    Acc_chord_idx = Acc_chord_idx.rename(columns={'Chord_idx_': 'Chord_idx', 'Correct_<lambda_0>': "Sample_Size", 
                                    'Correct_<lambda_1>': "Chord_Accuracy"})
    Acc_chord_idx = Acc_chord_idx.loc[Acc_chord_idx['Sample_Size']>=15]
    Acc_chord_idx = Acc_chord_idx.sort_values(by = 'Chord_Accuracy')
    return Acc_chord_idx


def target_seq_accuracy(result_table, train_chord_size_seq):
    target_seq_accuracy = result_table.groupby(
        by=["Target_Seq"]).agg({'Correct': [num_chords, correct_chords] }).reset_index()
    target_seq_accuracy.columns = ["_".join(x) for x in target_seq_accuracy.columns.values]
    target_seq_accuracy = target_seq_accuracy.rename(columns={'Target_Seq_': 'Target_Seq', 'Correct_<lambda_0>': "Seq_Sample_Size", 
                                    'Correct_<lambda_1>': "Seq_Accuracy"})
    target_seq_accuracy = target_seq_accuracy.merge(train_chord_size_seq, left_on = 'Target_Seq', 
                                                    right_on = 'Target_Seq_train',  how='left')
    # target_seq_accuracy = target_seq_accuracy.sort_values(by = 'Seq_Accuracy')
    target_seq_accuracy['Sample_Size_Train_Seq'].replace(np.nan, 0, inplace=True)

    return target_seq_accuracy


#%% Functions to extract information from targets and preds


def root_pitch(result_table, root_type, chord_type, mod_type):
    result_table[root_type] = 0
    result_table[root_type] = result_table[chord_type].str.slice(stop=1)
    result_table[mod_type] = 0
    result_table[mod_type] = result_table[chord_type].str.slice(start = 1, stop = 2)
    mods_to_keep = ['#', 'b']
    result_table['mod2'] = 0
    result_table.loc[ result_table[mod_type] == '#', "mod2"] = '#'
    result_table.loc[ result_table[mod_type] == 'b', "mod2"] = 'b'
    result_table.loc[ ~result_table[mod_type].isin(mods_to_keep), 'mod2'] = ""
    result_table[root_type] = result_table[root_type].str.cat(result_table['mod2'])
    result_table = result_table.drop(['mod2'], axis=1)
    # Add 'wrong extra' throw away column for wrong predictions
    # df.loc[(df['Correct'] == 0) & (df['p_root'] == df['t_root']), 'p_root'] = 'wrong extra'
    return result_table

def triad_vector(df, var, mod_vector, chord_info_name, triad_var, triad_var_corr, t_root, p_root ):
    mods_to_keep = ['#', 'b']
    df.loc[ ~df[mod_vector].isin(mods_to_keep), chord_info_name] = df[var].str.slice(start = 1)
    df.loc[ df[mod_vector].isin(mods_to_keep), chord_info_name] = df[var].str.slice(start = 2)
        
    df[triad_var] = 'M'    #initialise the triad form to major
    modes = ['o', '-', '+', 'sus'] #define other forms
    
    
    for i in modes:
        df.loc[df[var].str.contains(i, regex = False) == True, 
                  triad_var ] = i #fill the triad variable with the information in the chord mapping
    
    df.loc[df[var].str.contains('7b5', regex = False) == True, 
              triad_var ] = 'half' #Add the half-diminished form, only for 7b5 chords
    
    df[triad_var_corr] = df[triad_var]
    df.loc[ df[t_root] != df[p_root] , triad_var_corr] = 'wrong_root'

    return df



def added_note_vector(df, var, added_note_var, added_note_var_cor, t_triad, p_triad_corr):   

    df[added_note_var] = df[var]

    
    #remove all information which is already in the triad form vector 
    # (7alt --> mapped to 7)
    remove_note_info = ['-', '+', 'sus',  'b5'] 
    for i in remove_note_info:
        df[added_note_var] = df[added_note_var].str.replace(i,'', regex=True)
    
    # in the added note, diminished is only kept when it affects the 7th note
    # if only for the triad, already in the previous vector
    df.loc[ df[added_note_var] == 'o', added_note_var] = ''
    df.loc[ df[added_note_var] == '7', added_note_var] = 'm7'
    df.loc[df[added_note_var] == '', added_note_var] = 'none'
    
    df[added_note_var_cor] = df[added_note_var]
    df.loc[ df[t_triad]!=df[p_triad_corr], added_note_var_cor] = 'wrong_chord'
    
    return df

def separating_vectors_acc(df):
    result_table = df
    result_table = root_pitch(result_table, 't_root', 'Target_Chords', 't_mod')
    result_table = root_pitch(result_table,'p_root', 'Pred_Chords', 'p_mod')
    result_table = triad_vector(result_table, 'Target_Chords', 't_mod', 
                            'chord_info_T', 'triad_T', 'triad_T_corr', 
                            't_root', 'p_root' )
    result_table = triad_vector(result_table, 'Pred_Chords', 'p_mod', 
                            'chord_info_P', 'triad_P', 'triad_P_corr', 
                            't_root', 'p_root' ) 
    result_table = added_note_vector(result_table, 'chord_info_T', 'added_note_T', 'added_note_T_corr','triad_T', 'triad_P_corr')
    result_table = added_note_vector(result_table, 'chord_info_P', 'added_note_P', 'added_note_P_corr','triad_T', 'triad_P_corr')
    return result_table
    
#%% Creating confusion matrices

def create_save_matrix(df, var1, var2, plot_name, model_name, title=None, show=True, norm='all'):
    matrix = pd.crosstab(df[var1], df[var2], rownames=['Target'], colnames=['Predicted'], normalize=norm).round(4)*100
    if norm == 'index':
        matrix = matrix.round(0).astype(int)
    else:
         matrix = matrix.round(1)
    matrix_fig = sn.heatmap(matrix, annot=True)
    plt.title(title)
    if show:
        plt.show()
    else:
        plt.show(block=False)
        plt.close('all')
    
    # savepath = str(model_name) + str(plot_name)
    # figure = matrix_fig.get_figure()   
    # figure.savefig(savepath, dpi=400)


def get_decoded_melody(melid):
    dict = {
        0 : 'C',
        1 : 'C#',
        2 : 'D',
        3 : 'D#',
        4 : 'E',
        5 : 'F',
        6 : 'F#',
        7 : 'G',
        8 : 'G#',
        9 : 'A',
        10 : 'A#',
        11 : 'B'
    }
    seq_melody = get_dataset_multi_hot(choice=4, return_seq_melid=melid)
    # Remove -1s from sequences
    seq_melody = [[elem for elem in seq if elem != -1] for seq in seq_melody]
    decoded_mel = [list(pd.Series(seq, dtype='object').map(dict)) for seq in seq_melody]

    return decoded_mel

    
def song_analysis(df, song_id, model_name, model_name_mel):
    result_song = df.loc[df['Song']== song_id]
    result_song = root_pitch(result_song, 't_root', 'Target_Chords', 't_mod')
    result_song = root_pitch(result_song,'p_root', 'Pred_Chords', 'p_mod')
    result_song = root_pitch(result_song,'p_root_mel', 'Pred_Chords_mel', 'p_mod_mel')
    create_save_matrix(result_song, 't_root', 'Pred_Root',  str(song_id) + '_roots_crossmatrix.png', model_name, title=None, show=True, norm='index') #  norm='index' makes the normalization by rows. Delete if normalization by entire matrix is wanted
    create_save_matrix(result_song, 't_root', 'Pred_Root_mel', str(song_id) +'_roots_crossmatrix.png', model_name_mel, title=None, show=True, norm='index')
    create_save_matrix(result_song, 'triad_T', 'Pred_triad_form',  str(song_id) + '_triadform_crossmatrix.png', model_name, title=None, show=True)
    create_save_matrix(result_song, 'triad_T', 'Pred_triad_form_mel', str(song_id) + '_triadform_crossmatrix.png', model_name_mel, title=None, show=True)
    create_save_matrix(result_song, 'added_note_T', 'Pred_added_note',  str(song_id) + '_addednote_crossmatrix.png', model_name, title=None, show=True)
    create_save_matrix(result_song, 'added_note_T', 'Pred_added_note_mel', str(song_id) + '_addednote_crossmatrix.png', model_name_mel, title=None, show=True)
    create_save_matrix(result_song, 'Target_Chords', 'Pred_Chords', str(song_id) + '_crossmatrix.png', model_name, title=None, show=True)
    create_save_matrix(result_song, 'Target_Chords', 'Pred_Chords_mel', str(song_id) + '_crossmatrix.png', model_name_mel, title=None, show=True)
    result_song_simple = result_song[['Test_sample_ID', 'Song', 'Target_Chords', 'Pred_Chords', 'Pred_Chords_mel']]
    decoded_mel = get_decoded_melody(song_id)
    decoded_mel.pop(0)
    result_song_simple['Melody'] = decoded_mel
    # Check whether the specified path exists or not
    path = pathlib.os.path.join(result_analysis_path,'song_'+ str(song_id))
    if not os.path.exists(path):
        # Create a new directory because it does not exist 
        os.makedirs(path)
    result_song_simple.to_csv(pathlib.os.path.join(path, 'Model Predictions Compared.csv'), 
                      sep=';', header=True, index=False)
    return result_song


def compare_dominent_with_minor(targets, preds, preds_mel):
    total_dominent = 0
    correct_dominent_d1 = 0
    correct_dominent_d4 = 0
    counter_total_chords = 0
    pred_dominent_d1 = 0
    pred_dominent_d4 = 0
    for idx_song, song in enumerate(targets):
        if len(song) > 0:
            chords = song.values.flatten()
            for idx_chord, chord in enumerate(chords):
                if len(chord) <= 3 and not '-' in chord and not '+' in chord and not 'sus' in chord and not 'o' in chord and not 'sus' in chord and not 'j' in chord and not 'alt' in chord:
                    total_dominent += 1
                    if chord == preds[idx_song][0][idx_chord]:
                        correct_dominent_d1 += 1
                    if chord == preds_mel[idx_song][0][idx_chord]:
                        correct_dominent_d4 += 1
                    #print("Target: ", chord, ", Pred_d1: ", preds[idx_song][0][idx_chord], " Pred_d4: ", preds_mel[idx_song][0][idx_chord])
                    if '-' in preds[idx_song][0][idx_chord]:
                        pred_dominent_d1 += 1
                    if '-' in preds_mel[idx_song][0][idx_chord]:
                        pred_dominent_d4 += 1
                counter_total_chords += 1
    acc_dominent_d1 = correct_dominent_d1 / total_dominent
    acc_dominent_d4 = correct_dominent_d4 / total_dominent
    print("\nTotal chords: ", counter_total_chords)
    print("Accuracy dominent d1: ", acc_dominent_d1)
    print("Accuracy dominent d4: ", acc_dominent_d4)
    print("Number of wrongly predictated minor instead of dominent d1: ", pred_dominent_d1)
    print("Number of wrongly predictated minor instead of dominent d4: ", pred_dominent_d4)

#compare_dominent_with_minor(targets, preds, preds_mel)
#%% Creating the main tables

train_table = create_train_table(dataset, seed)
train_chord_size = train_sample_size_chords(train_table, size_seq = False)
train_chord_size_seq=train_sample_size_chords(train_table, size_seq = True)
song_acc_diff = compare_accuracies_table(song_list, song_length, song_accuracy,song_list_mel, song_length_mel, song_accuracy_mel)
song_acc_diff.to_csv('Tune Accuracy comparison.csv', sep=';', header=True, index=False)

result_table, results_numbers = create_result_table(song_list, song_length, song_accuracy, preds, targets)
result_table_mel, results_numbers_mel = create_result_table(song_list_mel, song_length_mel, song_accuracy_mel, preds_mel, targets_mel )

total_samples = len(result_table)
print('Baseline model:')
print('Test accuracy: %.2f' % (result_table.Correct.sum() / total_samples * 100))
print('Accuracy | Prev. correct: %.2f' % (result_table.correct_prev_c.sum() / result_table.Correct.sum() * 100))
print('Accuracy | Prev. incorrect: %.2f\n' % (result_table.correct_prev_inc.sum() / (total_samples - result_table.Correct.sum()) * 100))

print('Melody model:')
print('Test accuracy: %.2f' % (result_table_mel.Correct.sum() / len(result_table_mel) * 100))
print('Accuracy | Prev. correct: %.2f' % (result_table_mel.correct_prev_c.sum() / result_table_mel.Correct.sum() * 100))
print('Accuracy | Prev. incorrect: %.2f\n' % (result_table_mel.correct_prev_inc.sum() / (total_samples - result_table_mel.Correct.sum()) * 100))


resul_table_sep_vec = separating_vectors_acc(result_table)
resul_table_sep_vec_mel = separating_vectors_acc(result_table_mel)
result_table_all = resul_table_sep_vec[['Test_sample_ID', 'Chord_idx', 'Song', 'Target_Chords', 'Pred_Chords', 't_root', 'p_root', 'triad_T', 'triad_P_corr', 'added_note_T', 'added_note_P_corr']].merge(
    resul_table_sep_vec_mel[['Test_sample_ID', 'Chord_idx', 'Song', 'Target_Chords', 'Pred_Chords', 't_root', 'p_root', 'triad_T', 'triad_P_corr', 'added_note_T', 'added_note_P_corr']], left_on=['Test_sample_ID', 'Chord_idx', 'Song', 'Target_Chords', 't_root', 'triad_T',  'added_note_T'], 
    right_on = ['Test_sample_ID', 'Chord_idx', 'Song', 'Target_Chords', 't_root', 'triad_T',  'added_note_T'])
result_table_all = result_table_all.rename(columns = {'Pred_Chords_x' : 'Pred_Chords', 
                                                      'Pred_Chords_y' : 'Pred_Chords_mel',
                                                      'p_root_x' : 'Pred_Root', 
                                                      'p_root_y' : 'Pred_Root_mel',
                                                      'triad_P_corr_x' : 'Pred_triad_form',
                                                      'triad_P_corr_y' : 'Pred_triad_form_mel',
                                                      'added_note_P_corr_x' : 'Pred_added_note',
                                                      'added_note_P_corr_y' : 'Pred_added_note_mel'  })

if do_conf_matrix_all_songs:
    for i in result_table['Test_sample_ID'].unique():
        song_df = root_pitch(result_table[result_table['Test_sample_ID'] == i])
        confusion_matrix_root = pd.crosstab(song_df['t_root'], song_df['p_root'], rownames=['Target'], colnames=['Predicted'], normalize='all').round(4)*100
        title = 'Song test ID ' + str(i) + '. Length: ' + str(results_numbers.at[i, 'Song_Length']) + ' Acc: ' + str(results_numbers.at[i, 'Song_Accuracy'])
        create_save_matrix(confusion_matrix_root,  'roots_crossmatrix_song_' + str(i) + '.png', 'result_analysis/all_songs/', title=title, show=True)
        

        
#%% Analysis

F_score_chords = F_score(result_table, train_chord_size)

F_score_mel = F_score(result_table_mel, train_chord_size)


Acc_chord_idx = accuracy_chord_idx(result_table)
Acc_chord_idx_mel = accuracy_chord_idx(result_table_mel)

target_seq_accuracy_chords= target_seq_accuracy(result_table, train_chord_size_seq)
target_seq_accuracy_chords['model'] = 'Chords_Only'
target_seq_accuracy_mel = target_seq_accuracy(result_table_mel, train_chord_size_seq)
target_seq_accuracy_mel['model'] = 'Melody'

target_seq_all = pd.concat([target_seq_accuracy_chords, target_seq_accuracy_mel], axis = 0)
target_seq_all = target_seq_all.sort_values(by = 'Sample_Size_Train_Seq' )

#%%
song_id = 197
result_song = song_analysis(result_table_all, song_id, model_name, model_name_mel)

# histogram of test chords
# result_table['t_root'].value_counts().plot(kind='bar')
# plt.show()
#%%
create_save_matrix(result_table_all, 't_root', 'Pred_Root',  '_roots_crossmatrix.png', model_name, title=None, show=True, norm='index')  #  norm='index' makes the normalization by rows. Delete if normalization by entire matrix is wanted
create_save_matrix(result_table_all, 't_root', 'Pred_Root_mel',  '_roots_crossmatrix.png', model_name_mel, title=None, show=True, norm='index')

create_save_matrix(result_table_all, 'triad_T', 'Pred_triad_form',  '_triad_form_crossmatrix.png', model_name, title=None, show=True, norm='index')
create_save_matrix(result_table_all, 'triad_T', 'Pred_triad_form_mel',  '_triad_form_crossmatrix.png', model_name_mel, title=None, show=True, norm='index')

create_save_matrix(result_table_all, 'added_note_T', 'Pred_added_note',  '_added_note_crossmatrix.png', model_name, title=None, show=True, norm='index')
create_save_matrix(result_table_all, 'added_note_T', 'Pred_added_note_mel',  '_added_note_crossmatrix.png', model_name_mel, title=None, show=True, norm='index')
  

#%% Make plots
# x and y given as array_like objects

def scatter(df, x_var, y_var, title, xlabel, ylabel, size= None, hover_name = None, color = None, color_name= None):
    if color == None and size!=None:
        fig = px.scatter(df, x= x_var, y= y_var, size = size, hover_name= size,size_max=60,
                 labels={
                     x_var: xlabel,
                     y_var: ylabel},
                 title = title)
    if (color == None and  size == None):
        fig = px.scatter(df, x= x_var, y= y_var,
                 labels={
                     x_var: xlabel,
                     y_var: ylabel},
                 title = title)
    if (color != None and size != None):
        fig = px.scatter(df, x= x_var, y= y_var, color = color, size = size ,
                         hover_name= size,
                 labels={
                     x_var: xlabel,
                     y_var: ylabel,
                     color : color_name},
                 title = title)
    if  (color != None and size == None):
        fig = px.scatter(df, x= x_var, y= y_var, color = color, 
                 labels={
                     x_var: xlabel,
                     y_var: ylabel,
                     color : color_name},
                 title = title)
    fig.update_layout(
        font=dict(
            size=23))

    return fig 


#%%

pio.renderers.default='browser'

# "Chord F-score in the melody model weigthed by training chord appearance"
F_score_mel = F_score_mel.loc[F_score_mel['Sample_Size_x']>5]
plot_f_score = scatter(F_score_mel.sort_values(by = 'Target_Chords'),
                       'Target_Chords', 'f_score', None,
                       'Target Chords', 'F-score (%)', size = "Sample_Size_Train" )    

# "Chord accuracy vs chord index in the song sequence"
acc_chord_idx = scatter(Acc_chord_idx, 'Chord_idx', 'Chord_Accuracy', 
                        None,
                       'Chords Index', 'Chord Accuracy (%)', size = "Sample_Size" )  
# 'Chord accuracy vs chord sequence frequency in training'
target_seq_plot = scatter(target_seq_all.loc[target_seq_all['Seq_Sample_Size']>15], 
                          'Sample_Size_Train_Seq', 'Seq_Accuracy', 
                          None, 
                          'Chord sequence appearance in training', 
                          'Chord accuracy (%)', 
                          size = 'Seq_Sample_Size', 
                          hover_name = 'model',
                        color = 'model', 
                        color_name = 'Model Trained')  

acc_chord_idx.show()
plot_f_score.show()
target_seq_plot.show()