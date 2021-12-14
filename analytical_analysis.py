# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 16:05:52 2021

@author: nerea
"""

#%% Load Models
from argparse import ArgumentParser
from train import load_model
import numpy as np
import plotly.express as px
import plotly.io as pio

model_path = 'models/trained_models/model_trained_nerea.pth'
model_name = '3hot_chords_only'

parser = ArgumentParser(description='Train a model')
parser.add_argument('--load-path', type=str,
                    default= model_path,
                    help='')

args = parser.parse_args()
song_list, song_length, song_accuracy, preds, targets = load_model(args.load_path)

#%%
        
import pandas as pd

results_numbers = pd.DataFrame()
results_numbers['Songs'] = song_list
results_numbers['Song_Length'] = song_length
results_numbers['Song_Accuracy'] = song_accuracy
results_numbers['Test_sample_ID'] = range(1, 44)

        
#%%

result_table = pd.DataFrame(columns=['Test_sample_ID', 'Target_Chords', 'Pred_Chords'])

for i, j  in enumerate (targets):
    result_song = pd.DataFrame()
    result_song['Test_sample_ID'] = np.repeat(i, len(j))
    result_song['Chord_idx'] = range(1, len(j)+1)
    result_song['Target_Chords'] = j
    result_song['Pred_Chords'] = preds[i]
    result_table = pd.concat([result_table, result_song], axis=0)
    
    
result_table = result_table.merge(results_numbers, left_on='Test_sample_ID', right_on='Test_sample_ID')


chord_list = pd.unique(result_table['Target_Chords'])

result_table.loc[result_table['Target_Chords'] == result_table['Pred_Chords'], 'Correct'] = 1
result_table.loc[result_table['Target_Chords'] != result_table['Pred_Chords'], 'Correct'] = 0

# general functions

correct_chords = lambda x: sum(x)/len(x)*100
num_chords = lambda x: len(x)

Acc_chord_idx = result_table.groupby(
    by=["Chord_idx"]).agg({'Correct': [num_chords, 
                                           correct_chords] }).reset_index()
Acc_chord_idx.columns = ["_".join(x) for x in Acc_chord_idx.columns.ravel()]
Acc_chord_idx = Acc_chord_idx.rename(columns={'Chord_idx_': 'Chord_idx', 'Correct_<lambda_0>': "Sample_Size", 
                                'Correct_<lambda_1>': "Chord_Accuracy"})

Acc_chord_idx = Acc_chord_idx.loc[Acc_chord_idx['Sample_Size']>=15]



#%%
## precision is the number of true positive results divided by the number of all positive results,
 # including those not identified correctly,
 # and the recall is the number of true positive results divided 
 # by the number of all samples that should have been identified as positive
  # Precision is also known as positive predictive value,
  # and recall is also known as sensitivity in diagnostic binary classification.
  
# For prediction --> use the preds (corrects/total predicted)
# For recall use the targets

# 1- Compute recall 

Recall_Chords = result_table.groupby(
    by=["Target_Chords"]).agg({'Correct': [num_chords, 
                                           correct_chords] }).reset_index()
Recall_Chords.columns = ["_".join(x) for x in Recall_Chords.columns.ravel()]
Recall_Chords = Recall_Chords.rename(columns={'Target_Chords_': 'Target_Chords', 'Correct_<lambda_0>': "Sample_Size", 
                                'Correct_<lambda_1>': "recall"})

# 2- Compute prediction 


Prediction_Chords = result_table.groupby(
    by=["Pred_Chords"]).agg({'Correct': [num_chords, correct_chords] }).reset_index()
Prediction_Chords.columns = ["_".join(x) for x in Prediction_Chords.columns.ravel()]
Prediction_Chords = Prediction_Chords.rename(columns={'Pred_Chords_': 'Pred_Chords', 'Correct_<lambda_0>': "Sample_Size", 
                                'Correct_<lambda_1>': "precision"})

# 3- Merge the tables to get the F-score

# F_score = 2*((precision*recall)/(precision+recall))

F_score = Recall_Chords.merge(Prediction_Chords, left_on='Target_Chords', right_on='Pred_Chords')


F_score = F_score.assign(f_score = 2*((F_score['recall'] * F_score['precision'])/(F_score['recall'] + F_score['precision'])))
F_score['f_score'] = F_score['f_score'].replace(np.nan, 0)

F_score = F_score.sort_values(by = 'f_score')

#%% 
# Accuracy_Chord.columns = Accuracy_Chord.columns.droplevel(0)

prev_chord = ['None']
for i in range(1, len(result_table['Target_Chords'])):
    prev_chord.append(result_table['Target_Chords'][i-1])
    
result_table['Previous_target'] = prev_chord


#%% Make plots
# x and y given as array_like objects


Acc_chord_idx = Acc_chord_idx.sort_values(by = 'Chord_Accuracy')

import matplotlib.pyplot as plt
import seaborn as sn



pio.renderers.default='browser'
fig2 = px.scatter(result_table, x="Song_Length", y='Song_Accuracy')
fig3 = px.scatter(Acc_chord_idx, x="Chord_idx", y='Chord_Accuracy', 
                  size = 'Sample_Size')
fig4 = px.scatter(F_score, x = 'Target_Chords', y = 'f_score',
                  size="Sample_Size_x", hover_name="Target_Chords", size_max=60)

fig2.show()
fig3.show()
fig4.show()

#%%

result_table = result_table.assign(t_root = result_table['Target_Chords'].str.slice(stop=1))
result_table = result_table.assign(p_root = result_table['Pred_Chords'].str.slice(stop=1))

result_table = result_table.assign(t_mod = result_table['Target_Chords'].str.slice(start = 1, stop = 2))
result_table = result_table.assign(p_mod = result_table['Pred_Chords'].str.slice(start = 1, stop = 2))
mods_to_keep = ['#', 'b']
result_table['t_mod2'] = 0
result_table['p_mod2'] = 0
result_table.loc[ result_table['t_mod'] == '#', "t_mod2"] = '#'
result_table.loc[ result_table['t_mod'] == 'b', "t_mod2"] = 'b'
result_table.loc[ ~result_table['t_mod'].isin(mods_to_keep), 't_mod2'] = ""
result_table['t_root'] = result_table['t_root'].str.cat(result_table['t_mod2'])
result_table.loc[ result_table['p_mod'] == '#', "p_mod2"] = '#'
result_table.loc[ result_table['p_mod'] == 'b', "p_mod2"] = 'b'
result_table.loc[ ~result_table['p_mod'].isin(mods_to_keep), 'p_mod2'] = ""
result_table['p_root'] = result_table['p_root'].str.cat(result_table['p_mod2'])

confusion_matrix = pd.crosstab(result_table['t_root'], result_table['p_root'], rownames=['Actual'], colnames=['Predicted'], 
                               normalize='index').round(4)*100


matrix_fig = sn.heatmap(confusion_matrix, annot=True)
plt.show()

plot_name = '_roots_crossmatrix.png'

savepath = str(model_name) + str( plot_name)

figure = matrix_fig.get_figure()    
figure.savefig(savepath, dpi=400)
