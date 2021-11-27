# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:50:05 2021

@author: nerea
"""

from sqlalchemy import create_engine
import pandas as pd
import numpy as np



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

'''
Dataset 1: Basic dataset with only root pitches and '#'. Vocab size = 13. One-hot
'''
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

beats['Final_pitch'] = beats['complete_pitch']
beats.loc[ beats['complete_pitch'] == 'Db', "Final_pitch"] = 'C#'
beats.loc[ beats['complete_pitch'] == 'Eb', "Final_pitch"] = 'D#'
beats.loc[ beats['complete_pitch'] == 'Fb', "Final_pitch"] = 'E'
beats.loc[ beats['complete_pitch'] == 'Gb', "Final_pitch"] = 'F#'
beats.loc[ beats['complete_pitch'] == 'Ab', "Final_pitch"] = 'G#'
beats.loc[ beats['complete_pitch'] == 'Bb', "Final_pitch"] = 'A#'
beats.loc[ beats['complete_pitch'] == 'Cb', "Final_pitch"] = 'B'

''' Now only 12 pitches + No Chord, map it to the MIDI numeric notation for each pitch'''

conditions = [
    beats['Final_pitch']=='A#', 
    beats['Final_pitch']=='G', 
    beats['Final_pitch']=='C', 
    beats['Final_pitch']=='F', 
    beats['Final_pitch']=='D#', 
    beats['Final_pitch']=='G#', 
    beats['Final_pitch']=='D', 
    beats['Final_pitch']=='A', 
    beats['Final_pitch']=='F#', 
    beats['Final_pitch']=='B', 
    beats['Final_pitch']=='E', 
    beats['Final_pitch']=='C#', 
    beats['Final_pitch']=='N']
choices = [10, 7, 0, 5, 3, 8, 2, 9, 6, 11, 4, 1, 12]


# chord numbers have been set following MIDI notation, 
#without taking into acount octaves, minor/minor chords, neither the bass pitch

beats["chord_num"] = np.select(conditions, choices) 
beats = beats[['beatid', 'melid', 'chord', 'Final_pitch', 'chord_num', 'chord_info']] #remove useless columns



beats['chord_info'] = beats['chord_info'].str.replace('j','')
beats['chord_info'] = beats['chord_info'].str.replace('+','')
beats['chord_info'] = beats['chord_info'].str.replace('13','')
beats['chord_info'] = beats['chord_info'].str.replace('m','-')
beats['chord_info'] = beats['chord_info'].str.replace('11','')
beats['chord_info'] = beats['chord_info'].str.replace('b','')
beats['chord_info'] = beats['chord_info'].str.replace('#','')
beats['chord_info'] = beats['chord_info'].str.replace('9','')
beats['chord_info'] = beats['chord_info'].str.replace('\/(.*)','')
beats['chord_info'] = beats['chord_info'].str.replace('C','')

beats['chord_extra'] = beats['Final_pitch'].str.cat(beats['chord_info'])


beats['minor'] = '0'
beats.loc[beats['chord_info'].str.contains('-') == True, 'minor' ] = '1'

beats['chord_info'] = beats['chord_info'].str.replace('-','')

num_chords = pd.unique(beats['chord_extra'])
num_chord_info = pd.unique(beats['chord_info'])