# -*- coding: utf-8 -*-


from sqlalchemy import create_engine
import pandas as pd
import numpy as np

path = "C:/Users/nerea/OneDrive/Documentos/GitHub/ML_project2/Data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
engine = create_engine(f"sqlite:///{path}")

beats = pd.read_sql("beats", engine)

### THIS PART WAS ONLY DATA EXPLORING
#chords = pd.unique(beats['chord'])
#signature =  pd.unique(beats['signature'])
#form = pd.unique(beats['form'])
#pitch = pd.unique(beats['bass_pitch'])[0:25]

''' HERE IS TO DECIDE HOW WIDE SOULD OUR VOCABULARY BE DEPENDING ON THE CHORDS APPEARING LESS THAN X TIMES
BE CAREFUL AS SOME VERY BASIC CHORDS ARE ALMOST NOT USED (AS D), but the algorithm should predict it '''


l = beats['chord']
l.replace('', np.nan, inplace=True)
l.dropna(inplace=True)
hist = l.value_counts().plot(kind='bar')
chords_count = l.value_counts()
chords_count_keep = chords_count[(chords_count >=10)]
chords_count_discard = chords_count[(chords_count<10)]



### Here is the part to build the chord vocabulary

beats = beats[['beatid', 'chord']].dropna()
beats = beats.assign(root_pitch = beats['chord'].str.slice(stop=1))
# the first letter is always the root note, 
# but for the 12 basic pitches we also need sharp and flat modifications (second position)
beats = beats.assign(mod = beats['chord'].str.slice(start = 1, stop = 2))
mods_to_keep = ['#', 'b']
beats['mod2'] = 0
beats.loc[ beats['mod'] == '#', "mod2"] = '#'
beats.loc[ beats['mod'] == 'b', "mod2"] = 'b'
beats.loc[ ~beats['mod'].isin(mods_to_keep), 'mod2'] = ""

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

''' Now only 12 pithes + No Chord, map it to the MIDI numeric notation for each pitch'''

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
beats = beats[['beatid', 'chord', 'chord_num']] #remove useless columns

