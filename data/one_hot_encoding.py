from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from dataset import OneHot_VLDataset, MultiHot_VLDataset
import pdb

import sys
sys.path.append('../')


'''
path = "wjazzd.db" # REPLACE THIS WITH PATH TO FILE
engine = create_engine(f"sqlite:///{path}")
beats = pd.read_sql("beats", engine)
'''
### THIS PART WAS ONLY DATA EXPLORING
#chords = pd.unique(beats['chord'])
#signature =  pd.unique(beats['signature'])
#form = pd.unique(beats['form'])
#pitch = pd.unique(beats['bass_pitch'])[0:25]

''' HERE IS TO DECIDE HOW WIDE SOULD OUR VOCABULARY BE DEPENDING ON THE CHORDS APPEARING LESS THAN X TIMES
BE CAREFUL AS SOME VERY BASIC CHORDS ARE ALMOST NOT USED (AS D), but the algorithm should predict it '''

'''
l = beats['chord']
l.replace('', np.nan, inplace=True)
l.dropna(inplace=True)
hist = l.value_counts().plot(kind='bar')
chords_count = l.value_counts()
chords_count_keep = chords_count[(chords_count >=10)]
chords_count_discard = chords_count[(chords_count<10)]
'''


### Here is the part to build the chord vocabulary

def get_dataset1(beats):
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

    '''for the complete pitch we need the note and the flat/sharp modification'''
    beats['complete_pitch'] = beats['root_pitch'].str.cat(beats['mod2'])

    '''Now we need to consider that C# = Db --> map all 'b' to the equivalent '#' pitch'''

    beats['final_pitch'] = beats['complete_pitch']
    beats.loc[ beats['complete_pitch'] == 'Db', "final_pitch"] = 'C#'
    beats.loc[ beats['complete_pitch'] == 'Eb', "final_pitch"] = 'D#'
    beats.loc[ beats['complete_pitch'] == 'Fb', "final_pitch"] = 'E'
    beats.loc[ beats['complete_pitch'] == 'Gb', "final_pitch"] = 'F#'
    beats.loc[ beats['complete_pitch'] == 'Ab', "final_pitch"] = 'G#'
    beats.loc[ beats['complete_pitch'] == 'Bb', "final_pitch"] = 'A#'
    beats.loc[ beats['complete_pitch'] == 'Cb', "final_pitch"] = 'B'

    ''' Now only 12 pitches + No Chord, map it to the MIDI numeric notation for each pitch'''

    conditions = [
        beats['final_pitch']=='A#', 
        beats['final_pitch']=='G', 
        beats['final_pitch']=='C', 
        beats['final_pitch']=='F', 
        beats['final_pitch']=='D#', 
        beats['final_pitch']=='G#', 
        beats['final_pitch']=='D', 
        beats['final_pitch']=='A', 
        beats['final_pitch']=='F#', 
        beats['final_pitch']=='B', 
        beats['final_pitch']=='E', 
        beats['final_pitch']=='C#', 
        beats['final_pitch']=='N']
    choices = [10, 7, 0, 5, 3, 8, 2, 9, 6, 11, 4, 1, 12]

    # chord numbers have been set following MIDI notation, 
    #without taking into acount octaves, minor/minor chords, neither the bass pitch

    beats["chord_num"] = np.select(conditions, choices) 
    beats = beats[['beatid', 'melid', 'chord', 'chord_num']] #remove useless columns
    return beats


def get_dataset_one_hot(choice=1, test_split=0.2):
    '''
    Generate train and test dataset. Based on dataset choice
    choice:
        1: Basic dataset with only root pitches and '#'. Vocab size = 13. One-hot
    '''

    path = "data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
    engine = create_engine(f"sqlite:///{path}")
    beats_raw = pd.read_sql("beats", engine)

    if choice == 1:
        beats = get_dataset1(beats_raw)


    sequences = []
    num_mels = beats['melid'].max()
    vocab_size = beats['chord_num'].nunique()

    # for each song load its chord seq
    for i in range(1, num_mels+1):
        seq = beats.loc[beats['melid'] == i]['chord_num'].to_numpy()
        if len(seq) > 1:
            sequences.append(seq)

    # Split
    split_idx = int(len(sequences)*(1-test_split))
    train_seq = sequences[:split_idx]
    test_seq = sequences[split_idx:]

    train_dataset = OneHot_VLDataset(train_seq, vocab_size)
    test_dataset = OneHot_VLDataset(test_seq, vocab_size)

    return train_dataset, test_dataset, vocab_size



if __name__ == "__main__":
    get_dataset_one_hot()
    
