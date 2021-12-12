# -*- coding: utf-8 -*-
#%% HEATHER AND PATHS 
"""
Created on Sat Nov 27 16:50:05 2021

@author: nerea
"""

from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import pathlib 


project_path = "C:/Users/nerea/OneDrive/Documentos/GitHub/ML_project2"
path = "C:/Users/nerea/OneDrive/Documentos/GitHub/ML_project2/data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
engine = create_engine(f"sqlite:///{path}")
beats = pd.read_sql("beats", engine)



''' HERE IS TO DECIDE HOW WIDE SOULD OUR VOCABULARY BE DEPENDING ON THE CHORDS APPEARING LESS THAN X TIMES
BE CAREFUL AS SOME VERY BASIC CHORDS ARE ALMOST NOT USED (AS D), but the algorithm should predict it '''

"""
l = beats['chord']
l.replace('', np.nan, inplace=True)
l.dropna(inplace=True)
hist = l.value_counts().plot(kind='bar')
chords_count = l.value_counts()
chords_count_keep = chords_count[(chords_count >=10)]
chords_count_discard = chords_count[(chords_count<10)]
"""


#%%  Dataset 1: Basic dataset with only root pitches. Vocab size = 13. One-hot

beats['chord'].replace('', np.nan, inplace=True)
beats = beats[['beatid', 'melid', 'chord']].dropna()
beats = beats.assign(root_pitch = beats['chord'].str.slice(stop=1))

# the first letter is always the root note, 
# but for the 12 basic pitches we also need sharp and flat modifications (second position)
beats = beats.assign(mod = beats['chord'].str.slice(start = 1, stop = 2))
mods_to_keep = ['#', 'b']
beats['mod2'] = 0
beats.loc[ beats['mod'] == '#', "mod2"] = '#'
beats.loc[ beats['mod'] == 'b', "mod2"] = 'b'
beats.loc[ ~beats['mod'].isin(mods_to_keep), 'mod2'] = ""
beats.loc[ ~beats['mod'].isin(mods_to_keep), 'chord_info'] = beats['chord'].str.slice(start = 1)
beats.loc[ beats['mod'].isin(mods_to_keep), 'chord_info'] = beats['chord'].str.slice(start = 2)


'''for the complete pitch we need the note and the flat/sharp modification'''
beats['complete_pitch'] = beats['root_pitch'].str.cat(beats['mod2'])

'''Now we need to consider that C# = Db --> map all 'b' to the equivalent '#' pitch'''

beats['Root_pitch'] = beats['complete_pitch']
beats.loc[ beats['complete_pitch'] == 'Db', "Root_pitch"] = 'C#'
beats.loc[ beats['complete_pitch'] == 'Eb', "Root_pitch"] = 'D#'
beats.loc[ beats['complete_pitch'] == 'Fb', "Root_pitch"] = 'E'
beats.loc[ beats['complete_pitch'] == 'Gb', "Root_pitch"] = 'F#'
beats.loc[ beats['complete_pitch'] == 'Ab', "Root_pitch"] = 'G#'
beats.loc[ beats['complete_pitch'] == 'Bb', "Root_pitch"] = 'A#'
beats.loc[ beats['complete_pitch'] == 'Cb', "Root_pitch"] = 'B'

''' Now only 12 pitches + No Chord, map it to the MIDI numeric notation for each pitch'''



conditions = [
    beats['Root_pitch']=='A#', 
    beats['Root_pitch']=='G', 
    beats['Root_pitch']=='C', 
    beats['Root_pitch']=='F', 
    beats['Root_pitch']=='D#', 
    beats['Root_pitch']=='G#', 
    beats['Root_pitch']=='D', 
    beats['Root_pitch']=='A', 
    beats['Root_pitch']=='F#', 
    beats['Root_pitch']=='B', 
    beats['Root_pitch']=='E', 
    beats['Root_pitch']=='C#', 
    beats['Root_pitch']=='N']
choices = [10, 7, 0, 5, 3, 8, 2, 9, 6, 11, 4, 1, 12]


# chord numbers have been set following MIDI notation, 
#without taking into acount octaves, minor/minor chords, neither the bass pitch

beats["chord_num"] = np.select(conditions, choices) 
beats = beats[['beatid', 'melid', 'chord', 'Root_pitch', 'chord_num', 'chord_info']] #remove useless columns

#%%  Dataset 2: NOW WE ADD A SECOND VECTOR WITH ALL THE INFO ABOUT THE CHORD BUT THE ROOT PITCH

''' Remove or change useless information'''

beats['chord_map'] = beats['chord_info']
beats['chord_map'] = beats['chord_map'].str.replace('\/(.*)','')
beats['chord_map'] = beats['chord_map'].str.replace('m','-')
chord_info_importance = beats['chord_map'].value_counts()


# ''' Create 2 options depending if we keep the #/b 
# from the added notes higher than 7 '''


# beats['chord_map2'] = beats['chord_map']
old_added_notes = ['9#','9b', '9', '11b', '11#', '11', '13b', '13#', '13']
# note_choices = ['7#', '7b', '7', '7b', '7#', '7', '7b', '7#', '7']


for i in old_added_notes:
    beats['chord_map'] = beats['chord_map'].str.replace(i,'7') #map all to 7
    beats['chord_map'] = beats['chord_map'].str.replace('77','7') #to avoid repeated 7


# for i, j in zip(old_added_notes, note_choices):
#     beats['chord_map2'] = beats['chord_map2'].str.replace(i,j) #map to 7 but keep the flat/sharp info
#     beats['chord_map2'] = beats['chord_map2'].str.replace('77','7')


# #Remove repetitions
# beats['chord_map2'] = beats['chord_map2'].str.replace('7b7b','7b')
# beats['chord_map2'] = beats['chord_map2'].str.replace('7b7','7b')
# beats['chord_map2'] = beats['chord_map2'].str.replace('7#7#','7#')
# beats['chord_map2'] = beats['chord_map2'].str.replace('7#7','7#')

   #create unique vectors for each mapping
mapping1 = pd.unique(beats['chord_map']) 
#mapping2 = pd.unique(beats['chord_map2'])


beats['mapped_chord'] = beats['Root_pitch'].str.cat(beats['chord_map'])
#beats['mapped_chord2'] = beats['Root_pitch'].str.cat(beats['chord_map2'])

num_chords_mapping1 = pd.unique(beats['mapped_chord'])
#num_chords_mapping2 = pd.unique(beats['mapped_chord2'])

# Create chord dictionary only using C root pitch
beats['C'] = 'C'
beats['C_chords_ref'] = beats['C'].str.cat(beats['chord_info'])
beats['C_chord_map'] = beats['C'].str.cat(beats['chord_map'])
#beats['C_chord_map2'] = beats['C'].str.cat(beats['chord_map2'])

chord_dict = beats[['C_chords_ref', 'C_chord_map']]
chord_dict = chord_dict.drop_duplicates(subset=None, 
            keep='first', inplace=False, ignore_index=False)

chord_dict = chord_dict.sort_values(by = ['C_chord_map'])

# save the chord dictionary
chord_dict.to_csv(pathlib.os.path.join(project_path,'Chord_Dictionary.csv'), 
                  sep=';', header=True, index=False)

#%%  Dataset 3: 3 VECTORS, ONE FOR THE ROOT PITCH, A SECOND FOR THE TRIAD FORM AND A THIRD FOR ADDED NOTES

'''The 2 vectors will be created using the first mapping of the previous dataset.'''    

''' TRIAD VECTOR '''
beats['triad'] = 'M'    #initialise the triad form to major
modes = ['o', '-', '+', 'sus'] #define other forms


for i in modes:
    beats.loc[beats['chord_map'].str.contains(i, regex = False) == True, 
              'triad' ] = i #fill the triad variable with the information in the chord mapping

beats.loc[beats['chord_map'].str.contains('7b5', regex = False) == True, 
          'triad' ] = 'half' #Add the half-diminished form, only for 7b5 chords

total_mode = pd.unique(beats['triad'])


''' ADDED NOTE VECTOR '''
    
beats['added_note'] = beats['chord_map']

#remove all information which is already in the triad form vector 
# (7alt --> mapped to 7)
remove_note_info = ['-', '+', 'sus',  'b5', 'alt'] 
for i in remove_note_info:
    beats['added_note'] = beats['added_note'].str.replace(i,'')

# 'C' from  No Chord is removed
beats['added_note'] = beats['added_note'].str.replace('C','')

# in the added note, diminished is only kept when it affects the 7th note
# if only for the triad, already in the previous vector
beats.loc[ beats['added_note'] == 'o', "added_note"] = ''
beats.loc[ beats['added_note'] == '7', "added_note"] = 'm7'
beats.loc[beats['added_note'] == '', 'added_note'] = 'none'


add_note_count = beats['added_note'].value_counts()


total_added_note = pd.unique(beats['added_note'])

#Save the 3hot encoding 
mapping3 = beats[['chord_info', 'triad', 'added_note' ]]
mapping3 = mapping3.drop_duplicates(subset=None, 
            keep='first', inplace=False, ignore_index=False)

mapping3 = mapping3.astype({"chord_info":'category', "triad":'category', 
                            'added_note':'category'})
mapping3.to_csv(pathlib.os.path.join(project_path,'3hot.csv'), 
                  sep=';', header=True,  
                  index=False)



#%% SAVE MODIFIED BEATS TABLE WITH INFORMATION FOR ALL DATASETS

beats = beats[['beatid', 'melid', 'chord', 'Root_pitch', 'chord_info', 
               'chord_map', 'triad', 'added_note']]

beats.to_csv(pathlib.os.path.join(project_path,'Beats_Modified.csv'), 
                  sep=';', header=True, index=False)
