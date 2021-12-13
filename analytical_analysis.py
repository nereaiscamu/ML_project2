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


if __name__ == "__main__":
    np.random.seed(42)

    parser = ArgumentParser(description='Train a model')
    parser.add_argument('--load-path', type=str,
                        # required=True,
                        #default=None,
                        default='models/trained_models/model_1_dataset_1_s42.pth',
                        help='')

    args = parser.parse_args()

    if args.load_path is not None:
        s, l, a, p, t = load_model(args.load_path)
        
import pandas as pd

results_numbers = pd.DataFrame()
results_numbers['Songs'] = s
results_numbers['Song_Length'] = l
results_numbers['Song_Accuracy'] = a
results_numbers['Test_sample_ID'] = range(1, 44)

        
#%%

result_table = pd.DataFrame(columns=['Test_sample_ID', 'Target_Chords', 'Pred_Chords'])

for i in range (1,len(t)):
    result_song = pd.DataFrame()
    result_song['Test_sample_ID'] = np.repeat(i, len(t[i]))
    result_song['Target_Chords'] = t[i]
    result_song['Pred_Chords'] = p[i]
    result_table = pd.concat([result_table, result_song], axis=0)
    
    
result_table = result_table.merge(results_numbers, left_on='Test_sample_ID', right_on='Test_sample_ID')


chord_list = pd.unique(result_table['Target_Chords'])

result_table.loc[result_table['Target_Chords'] == result_table['Pred_Chords'], 'Correct'] = 1
result_table.loc[result_table['Target_Chords'] != result_table['Pred_Chords'], 'Correct'] = 0

correct_chords = lambda x: sum(x)/len(x)*100
num_chords = lambda x: len(x)

Accuracy_Chord = pd.DataFrame(columns=['Target_Chords', 'Sample_Size', 'Chord_Accuracy'])
Accuracy_Chord = result_table.groupby(
    by=["Target_Chords"]).agg({'Correct': [num_chords, 
                                           correct_chords] }).reset_index()
Accuracy_Chord.columns = ["_".join(x) for x in Accuracy_Chord.columns.ravel()]
Accuracy_Chord = Accuracy_Chord.rename(columns={'Correct_<lambda_0>': "Sample_Size", 
                                'Correct_<lambda_1>': "Chord_Accuracy"})


#%%

# Accuracy_Chord.columns = Accuracy_Chord.columns.droplevel(0)




prev_chord = ['None']
for i in range(1, len(result_table['Target_Chords'])):
    prev_chord.append(result_table['Target_Chords'][i-1])
    
result_table['Previous_target'] = prev_chord

#%% Make plots
# x and y given as array_like objects

pio.renderers.default='browser'
fig = px.scatter(result_table, x= ('Correct', '<lambda_0>'), y=('Correct', '<lambda_1>'))
fig.show()

fig2 = px.scatter(Accuracy_Chord, x="Song_Length", y='Song_Accuracy')

    

