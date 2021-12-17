# -*- coding: utf-8 -*-
"""
Created on Fri Dec 17 09:26:26 2021

@author: nerea
"""

#%% Import libraries

from sqlalchemy import create_engine
from combine_melody_beats import encode_melody
import pandas as pd
import numpy as np
from dataset import OneHot_VLDataset, MultiHot_VLDataset, MultiHot_MelodyEncoded_VLDataset, MultiHot_MelodyBassEncoded_VLDataset, MultiHot_MelodyDurationEncoded_VLDataset, MultiHot_MelodyWeighted_VLDataset
import pdb
import pickle
import os
import sys
sys.path.append('../')

path = "./data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
engine = create_engine(f"sqlite:///{path}")
beats = pd.read_sql("beats", engine)
melody= pd.read_sql("melody", engine)


#%%

### Here is the part to build the chord vocabulary


def preprocess_chords(beats, mel_included = False):
    # Remove empty rows, if melody encoded, empty rows corresponds to -1
    if mel_included == False:
        beats= beats.loc[beats['chord'] != 'NC']
        beats['chord'].replace('', np.nan, inplace=True)
        # beats['chord'].replace(-1, np.nan, inplace=True)
        beats = beats.loc[~beats['chord'].isna()]
        
    if mel_included == True:
        beats = beats.replace({'': np.nan})
        beats['melid2'] = beats['melid']
        beats['chord'] = beats.groupby('melid2')['chord'].transform(lambda v: v.ffill())
        beats= beats.loc[beats['chord'] != 'NC']
        beats = beats.loc[~beats['chord'].isna()]
        
        
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

    # *** Now only 12 pitches + No Chord ***

    #beats = beats[['beatid', 'melid', 'chord', 'Root_pitch', 'chord_info', 'bar', 'beat', 'bass_pitch']] #remove useless columns
    beats = beats.drop(['mod', 'mod2', 'complete_pitch', 'root_pitch'], axis=1)

    # NOW WE ADD A SECOND VECTOR WITH ALL THE INFO ABOUT THE CHORD BUT THE ROOT PITCH

    ''' Remove or change useless information'''

    beats['chord_map'] = beats['chord_info']
    beats['chord_map'] = beats['chord_map'].str.replace('\/(.*)','')
    beats['chord_map'] = beats['chord_map'].str.replace('m','-')
    beats['chord_map'] = beats['chord_map'].str.replace('7alt','7')
    
    old_added_notes = ['9#','9b', '9', '11b', '11#', '11', '13b', '13#', '13']
    
    # When encoding the melody, type gets changed, need to change type to prevent error for ~ operator in beats.loc[~...
    beats['chord_map'] = beats['chord_map'].astype(str)

    for i in old_added_notes:
        beats.loc[beats['chord_map'].str.contains('6', regex = False) == True, 
                  'chord_map' ] = beats['chord_map'].str.replace(i,'')
        beats.loc[~beats['chord_map'].str.contains('6', regex = False) == True, 
                  'chord_map' ] = beats['chord_map'].str.replace(i,'7')
        beats['chord_map'] = beats['chord_map'].str.replace('77','7') #to avoid repeated 7
        
    beats['new_chord'] = beats['Root_pitch'].str.cat(beats['chord_map'])

    
    #  Dataset 3: 3 VECTORS, ONE FOR THE ROOT PITCH, A SECOND FOR THE TRIAD FORM AND A THIRD FOR ADDED NOTES
    
    '''The 2 vectors will be created using the first mapping of the previous dataset.'''    
    
    ''' TRIAD VECTOR '''
    beats['triad'] = 'M'    #initialise the triad form to major
    modes = ['o', '-', '+', 'sus'] #define other forms
    
    
    for i in modes:
        beats.loc[beats['chord_map'].str.contains(i, regex = False) == True, 
                  'triad' ] = i #fill the triad variable with the information in the chord mapping
    
    beats.loc[beats['chord_map'].str.contains('7b5', regex = False) == True, 
              'triad' ] = 'half' #Add the half-diminished form, only for 7b5 chords
        
    
    ''' ADDED NOTE VECTOR '''
        
    beats['added_note'] = beats['chord_map']
    
    #remove all information which is already in the triad form vector 
    # (7alt --> mapped to 7)
    remove_note_info = ['-', '+', 'sus',  'b5'] 
    for i in remove_note_info:
        beats['added_note'] = beats['added_note'].str.replace(i,'')
    
    # 'C' from  No Chord is removed
    beats['added_note'] = beats['added_note'].str.replace('C','')
    
    # in the added note, diminished is only kept when it affects the 7th note
    # if only for the triad, already in the previous vector
    beats.loc[ beats['added_note'] == 'o', "added_note"] = ''
    beats.loc[ beats['added_note'] == '7', "added_note"] = 'm7'
    beats.loc[beats['added_note'] == '', 'added_note'] = 'none'
    
    # beats.loc[beats['chord_info'] == 'C', 'triad'] = 'none'
    # beats.loc[beats['chord_info'] == 'C', 'chord_info'] = 'No Chord'
        
    # 'triad' contains 6 elements. 'added_note' contains 6 elements. Option 2: Root pitch (13) + triad (6) + extra_node (6)

    return beats

def concat_melody_beats3(df_melody, df_beats):
    """
    Uses the output of the preprocess_chords for mel_included = True

    Args:
        df_melody: Dataframe of the melody table
        df_beats: Dataframe of the beats table (after preprocessing)

    Returns:
        Dataframe consisting of the combination of both tables
    """
    # Remove useless columns
    df_melody = df_melody[['eventid', 'melid', 'pitch', 'duration', 'bar', 'beat']]

    # Define new index with the key (melid, bar, beat)
    new_index = ['melid', 'bar', 'beat']
    df_chords_new = df_beats.set_index(new_index, drop=True)
    df_melody_new = df_melody.set_index(new_index, drop=True)

    # Concatenate the dataframes using the new index and then reset the index again
    df_beats_mel = df_chords_new.merge(df_melody_new, left_on=new_index, right_on=new_index, how='left')
    df_beats_mel = df_beats_mel.reset_index(drop=False)

    return df_beats_mel

def encode_pitch(df_melody, df_beats, pitch_sequence=False):
    """
    Encodes the pitches of the melody in combination with the beats.

    Args:
        df_melody: Dataframe of the melody table
        df_beats: Dataframe of the beats table
        pitch_sequence: Boolean to use pitch per chord (False) or sequence of chord (True) 

    Returns:
        Dataframe containing the encoded pitches
    """

    df_beats_mel = concat_melody_beats3(df_melody, df_beats)
    
    # Encode pitch
    df_beats_mel['pitch_encoded'] = np.mod(df_beats_mel['pitch'], 12)
    df_beats_mel['bass_pitch_encoded'] = np.mod(df_beats_mel['bass_pitch'], 12)

    df_beats_mel.fillna(-1, inplace=True)
    df_beats_mel['pitch_encoded'] = df_beats_mel['pitch_encoded'].astype(int)
    max_pitch = df_beats_mel['pitch_encoded'].max()
    df_beats_mel['bass_pitch_encoded'] = df_beats_mel['bass_pitch_encoded'].astype(int)
    df_beats_mel['duration'] = df_beats_mel['duration'].round(4)

    ## Encode pitch for every chord of melody
    if not pitch_sequence:
        return df_beats_mel

    ## Encode sequence of pitch for every chord
    # Add column that represent chord changes
    df_beats_mel['chord_changed'] = (df_beats_mel['chord'].shift() != df_beats_mel["chord"]).cumsum()

    # Group chord changes to get sequences
    pitch_sequences = [g['pitch_encoded'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]
    bass_pitch_sequences = [g['bass_pitch_encoded'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]
    duration_sequence = [g['duration'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]

    # Identify last row of current chord
    df_beats_mel['pitch_sequence'] = (df_beats_mel['chord'].shift(-1) != df_beats_mel["chord"])

    # Change type to type object to add list to cell
    df_beats_mel['pitch_sequence'] = df_beats_mel['pitch_sequence'].astype(object)
    df_beats_mel['bass_pitch_sequence'] = df_beats_mel['pitch_sequence']
    df_beats_mel['duration_sequence'] = df_beats_mel['pitch_sequence']

    # Set sequence to last chord
    # TODO decrease running time if possible
    for idx, _ in df_beats_mel.iterrows():
        if df_beats_mel.at[idx, 'pitch_sequence'] == True:
            df_beats_mel.at[idx, 'pitch_sequence'] = pitch_sequences.pop(0)
            df_beats_mel.at[idx, 'bass_pitch_sequence'] = bass_pitch_sequences.pop(0)
            df_beats_mel.at[idx, 'duration_sequence'] = duration_sequence.pop(0)

    # melody_encoded = [np.zeros(max_pitch) for _ in range(len(df_beats_mel))]
    # df_beats_mel['melody_encoded'] = melody_encoded
    df_beats_mel.drop(['bass_pitch_encoded', 'chord_changed'], axis=1, inplace=True)
    df_beats_mel.drop(df_beats_mel[df_beats_mel['pitch_sequence'] == False].index, inplace=True)

    return df_beats_mel


#%%

def encode_chords_1(table):
    unique_chords = pd.unique(table['new_chord'])
    unique_pitch = pd.unique(table['Root_pitch'])
    unique_triad = pd.unique(table['triad'])
    unique_added_note = pd.unique(table['added_note'])
    
    # Encode each final pitch to a number by order of appearnce
    Root_pitch_map = {}
    for i, c in enumerate(unique_pitch):
        Root_pitch_map[c] = i
    table['Root_pitch_num'] = table['Root_pitch'].map(Root_pitch_map)

    # Encode extra note
    triad_map = {}
    for i, c in enumerate(unique_triad):
        triad_map[c] = i
    table['triad_num'] = table['triad'].map(triad_map)

    # Encode extra note
    added_note_map = {}
    for i, c in enumerate(unique_added_note):
        added_note_map[c] = i
    table['added_note_num'] = table['added_note'].map(added_note_map)

    # Encode new chord
    new_chord_map = {}
    for i, c in enumerate(unique_chords):
        new_chord_map[c] = i
    table['new_chord_num'] = table['new_chord'].map(new_chord_map)
    
    a_file = open("models/new_chord_map.pkl", "wb")
    pickle.dump(new_chord_map, a_file)
    a_file.close()

    return table

def encode_chords_2(table):
    unique_chords = pd.unique(table['new_chord'])
    unique_pitch = pd.unique(table['Root_pitch'])
    unique_chord_info = pd.unique(table['chord_map'])
    #vocab_sizes = [len(unique_pitch), len(unique_chord_info)]
    
    # Encode each final pitch to a number by order of appearnce
    Root_pitch_map = {}
    for i, c in enumerate(unique_pitch):
        Root_pitch_map[c] = i
    table['Root_pitch_num'] = table['Root_pitch'].map(Root_pitch_map)

    # Encode chord map
    chord_info_map = {}
    for i, c in enumerate(unique_chord_info):
        chord_info_map[c] = i
    table['chord_info_num'] = table['chord_map'].map(chord_info_map)

    # Encode new chord
    new_chord_map = {}
    for i, c in enumerate(unique_chords):
        new_chord_map[c] = i
    table['new_chord_num'] = table['new_chord'].map(new_chord_map)
    


    return table



#%% 

''' Defining datasets '''

def get_dataset_only_chord_1(beats):
    '''
    Dataset 2: Multi-hot. Combination of 3 One-Hot vectors:
        (1): Root pitch and '#'. Vocab size = 13
        (2): triad. Vocab size = 6
        (3): Extra note. Vocab size = 4
    '''
    #beats['chord'].replace('', np.nan, inplace=True)
    beats = beats[['beatid', 'melid', 'chord', 'bar', 'beat', 'bass_pitch']]
    beats = preprocess_chords(beats, mel_included = False)
    beats = encode_chords_1(beats)

    unique_chords = pd.unique(beats['new_chord'])
    unique_pitch = pd.unique(beats['Root_pitch'])
    unique_triad = pd.unique(beats['triad'])
    unique_added_note = pd.unique(beats['added_note'])
    vocab_sizes = [len(unique_pitch), len(unique_triad), len(unique_added_note)]
    target_size = len(unique_chords)

    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): triad. Vocab size of %d' % len(unique_triad))
    print('\t(3): Extra note. Vocab size of %d\n' % len(unique_added_note))

    return beats, vocab_sizes, target_size

def get_dataset_only_chord_2(beats):
    '''
    Dataset 2: Multi-hot. Combination of 3 One-Hot vectors:
        (1): Root pitch and '#'. Vocab size = 13
        (2): Chord info. Vocab size = 14
    '''
    # beats['chord'].replace('', np.nan, inplace=True)
    beats = beats[['beatid', 'melid', 'chord']]
    beats = preprocess_chords(beats, mel_included = False)

    beats = encode_chords_2(beats)

    unique_chords = pd.unique(beats['new_chord'])
    unique_pitch = pd.unique(beats['Root_pitch'])
    unique_chord_info = pd.unique(beats['chord_map'])
    vocab_sizes = [len(unique_pitch), len(unique_chord_info)]
    target_size = len(unique_chords)

    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Chord info. Vocab size of %d\n' % len(unique_chord_info))

    return beats, vocab_sizes, target_size




def get_dataset4(melody, beats):
    """
    Dataset 4: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Whether or not it is minor
        (3): Other chord info
        (4): Melody encoding. All notes combined into one embedding
    """
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    mel_beats = encode_pitch(melody, beats, pitch_sequence=True)
    mel_beats = preprocess_chords(mel_beats, mel_included = True)
    mel_beats = encode_chords_1(mel_beats)

    unique_chords = pd.unique(mel_beats['new_chord'])
    unique_pitch = pd.unique(mel_beats['Root_pitch'])
    unique_triad = pd.unique(beats['triad'])
    unique_added_note = pd.unique(beats['added_note'])
    unique_notes = pd.unique(mel_beats['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_triad), len(unique_added_note)]
    target_size = len(unique_chords)

    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): triad. Vocab size of %d' % len(unique_triad))
    print('\t(3): Extra note. Vocab size of %d' % len(unique_added_note))
    print('\t(4): Melody pitch. Vocab size of %d\n' % len(unique_notes))

    return mel_beats, vocab_sizes, target_size


def get_dataset5(melody, beats):
    """
    Dataset 4: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Whether or not it is minor
        (3): Other chord info
        (4): Melody encoding. All notes combined into one embedding
        (5): Bass pitch encoding
    """
    
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]



    beats_mel = encode_melody(melody, beats, pitch_sequence=True)    
    beats_mel = preprocess_chords(beats_mel, mel_included = True)
    beats_mel = encode_chords_1(beats_mel)
    
    
    unique_chords = pd.unique(beats_mel['new_chord'])
    unique_pitch = pd.unique(beats_mel['Root_pitch'])
    unique_triad = pd.unique(beats_mel['triad'])
    unique_added_note = pd.unique(beats_mel['added_note'])
    unique_notes = pd.unique(beats_mel['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_triad), len(unique_added_note)]
    target_size = len(unique_chords)
    
    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): triad. Vocab size of %d' % len(unique_triad))
    print('\t(3): Extra note. Vocab size of %d' % len(unique_added_note))
    print('\t(4): Melody pitch. Vocab size of %d\n' % len(unique_notes))

    return beats_mel, vocab_sizes, target_size


#%%
def get_dataset_multi_hot(choice=1, val_split=0.1, test_split=0.1, seed=42):
    '''
    Generate train and test dataset. Based on dataset choice
    choice:
        2: Multi-hot chord encoding
        3: Multi-hot chord encoding + melody encoding (event for every note)
        4: Multi-hot chord encoding + melody encoding (notes encoding combined)
        5: Multi-hot chord encoding + bass pitch encoding
        6: Multi-hot chord encoding + melody encoding & bass pitch encoding
        7: Multi-hot chord encoding + melody encoding with weighted duration
        8: Multi-hot chord encoding + weighted melody encoding
    '''

    path = "./data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
    engine = create_engine(f"sqlite:///{path}")
    beats_raw = pd.read_sql("beats", engine)
    melody_raw = pd.read_sql("melody", engine)

    if choice == 1:
        beats, vocab_sizes, target_size = get_dataset_only_chord_1(beats_raw)
    if choice == 2:
        beats, vocab_sizes, target_size = get_dataset_only_chord_2(beats_raw)
    if choice == 4:
        beats, vocab_sizes, target_size = get_dataset4(melody_raw, beats_raw)
    if choice == 5 or choice == 6 or choice == 7 or choice== 8:
        beats, vocab_sizes, target_size = get_dataset5(melody_raw, beats_raw)

    sequences = []          # store chord as multi-hot
    target_sequence = []    # store chord as one-hot
    num_mels = beats['melid'].max()
    print(num_mels)
    melodies = []
    bass_pitch = []

    # for each song load its chord seq
    for i in range(1, num_mels+1):
        song = beats.loc[beats['melid'] == i]
        seq_pitch = song['Root_pitch_num'].to_numpy()
        seq_one_hot = song['new_chord_num'].to_numpy()

        if choice == 2:
            seq_chord_info = song['chord_info_num'].to_numpy()
        else:
            seq_triad = song['triad_num'].to_numpy()
            seq_added_note = song['added_note_num'].to_numpy()

        if len(seq_pitch) > 1:
            # One input for each melody note
            if choice == 3: 
                seq_pitch_mel = song['pitch_encoded'].to_numpy()
                sequences.append(np.array([seq_pitch, seq_chord_info, seq_pitch_mel]).T)

            # Melody embedding includes all notes played during chord
            else:  
                if choice == 2:
                    sequences.append(np.array([seq_pitch, seq_chord_info]).T)
                else:
                    sequences.append(np.array([seq_pitch, seq_triad, seq_added_note]).T)
                    if choice == 4 or choice == 8:     # melody encoding
                        melody_encoding = song['pitch_sequence']
                        melodies.append(np.array(melody_encoding))
                    if choice == 5:     # bass encoding
                        bass_pitch_encoding = song['bass_pitch_sequence']
                        bass_pitch.append(np.array(bass_pitch_encoding))
                    if choice == 6:     # melody + bass encoding
                        melody_encoding = song['pitch_sequence']
                        bass_pitch_encoding = song['bass_pitch_sequence']
                        melodies.append(np.array(melody_encoding))
                        bass_pitch.append(np.array(bass_pitch_encoding))
                    if choice == 7:     # melody weighted with the duration of the notes
                        duration_melody = song['duration_sequence']
                        melody_encoding = song['pitch_sequence']
                        # Append a tuple of the melody and duration
                        melodies.append(np.array([(melody_encoding, duration_melody)], dtype='object, object'))


            target_sequence.append(seq_one_hot)
        else:
            print('len(seq_pitch) = %d, for melid: %d' % (len(seq_pitch), i))
            print(seq_pitch)
            
    # convert to np array
    sequences = np.array(sequences, dtype=object)
    target_sequence = np.array(target_sequence, dtype=object)
    melodies = np.array(melodies, dtype=object)
    bass_pitch = np.array(bass_pitch, dtype=object)

    print(len(sequences))
    print(len(target_sequence))

    # Split Train/Val/Test
    random_idxs = np.random.RandomState(seed=seed).permutation(len(sequences))    # this randomState has a localized effect, so the permutation will be the same always (and can use test set in load_model)
    split_1 = int(len(sequences)*(1-test_split-val_split))
    split_2 = int(len(sequences)*(1-test_split))

    train_idxs = random_idxs[:split_1]
    val_idxs = random_idxs[split_1:split_2]
    test_idxs = random_idxs[split_2:]

    train_seq = sequences[train_idxs]
    val_seq = sequences[val_idxs]
    test_seq = sequences[test_idxs]
    train_target_seq = target_sequence[train_idxs]
    val_target_seq = target_sequence[val_idxs]
    test_target_seq = target_sequence[test_idxs]

    if choice == 4:
        train_mel = melodies[train_idxs]
        val_mel = melodies[val_idxs]
        test_mel = melodies[test_idxs]
        train_dataset = MultiHot_MelodyEncoded_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
        val_dataset = MultiHot_MelodyEncoded_VLDataset(val_seq, val_mel, val_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyEncoded_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12

        # save datasets
        data = (train_dataset, val_dataset, test_dataset, input_size, target_size)
        with open('data/datasets/dataset4.pickle', 'wb') as f:
            pickle.dump(data, f)

        return train_dataset, val_dataset, test_dataset, input_size, target_size

    if choice == 5:
        train_bass = bass_pitch[train_idxs]
        val_bass = bass_pitch[val_idxs]
        test_bass = bass_pitch[test_idxs]
        train_dataset = MultiHot_MelodyEncoded_VLDataset(train_seq, train_bass, train_target_seq, vocab_sizes)
        val_dataset = MultiHot_MelodyEncoded_VLDataset(val_seq, val_bass, val_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyEncoded_VLDataset(test_seq, test_bass, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return train_dataset, val_dataset, test_dataset, input_size, target_size

    if choice == 6:
        train_mel = melodies[train_idxs]
        val_mel = melodies[val_idxs]
        test_mel = melodies[test_idxs]
        train_bass = bass_pitch[train_idxs]
        val_bass = bass_pitch[val_idxs]
        test_bass = bass_pitch[test_idxs]
        train_dataset = MultiHot_MelodyBassEncoded_VLDataset(train_seq, train_mel, train_bass, train_target_seq, vocab_sizes)
        val_dataset = MultiHot_MelodyBassEncoded_VLDataset(val_seq, val_mel, val_bass, val_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyBassEncoded_VLDataset(test_seq, test_mel, test_bass, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12 + 12
        return train_dataset, val_dataset, test_dataset, input_size, target_size

    if choice == 7:
        train_mel = melodies[train_idxs]
        val_mel = melodies[val_idxs]
        test_mel = melodies[test_idxs]     
        train_dataset = MultiHot_MelodyDurationEncoded_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
        val_dataset = MultiHot_MelodyDurationEncoded_VLDataset(val_seq, val_mel, val_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyDurationEncoded_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return train_dataset, val_dataset, test_dataset, input_size, target_size

    if choice == 8:
        train_mel = melodies[train_idxs]
        val_mel = melodies[val_idxs]
        test_mel = melodies[test_idxs]       
        train_dataset = MultiHot_MelodyWeighted_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
        val_dataset = MultiHot_MelodyWeighted_VLDataset(val_seq, val_mel, val_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyWeighted_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return train_dataset, val_dataset, test_dataset, input_size, target_size

    train_dataset = MultiHot_VLDataset(train_seq, train_target_seq, vocab_sizes)
    val_dataset = MultiHot_VLDataset(val_seq, val_target_seq, vocab_sizes)
    test_dataset = MultiHot_VLDataset(test_seq, test_target_seq, vocab_sizes)

    input_size = sum(vocab_sizes)


    # save datasets
    data = (train_dataset, val_dataset, test_dataset, input_size, target_size)
    with open('data/datasets/dataset1.pickle', 'wb') as f:
        pickle.dump(data, f)

    return train_dataset, val_dataset, test_dataset, input_size, target_size


def get_dataset_multi_hot_without_split(choice=1, test_split=0.1):
    '''
    Generate train and test dataset. Based on dataset choice
    choice:
        2: Multi-hot chord encoding
        3: Multi-hot chord encoding + melody encoding (event for every note)
        4: Multi-hot chord encoding + melody encoding (notes encoding combined)
        5: Multi-hot chord encoding + bass pitch encoding
        6: Multi-hot chord encoding + melody encoding & bass pitch encoding
        7: Multi-hot chord encoding + melody encoding with weighted duration
        8: Multi-hot chord encoding + weighted melody encoding
    '''

    path = "./data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
    engine = create_engine(f"sqlite:///{path}")
    beats_raw = pd.read_sql("beats", engine)
    melody_raw = pd.read_sql("melody", engine)

    if choice == 1:
        beats, vocab_sizes, target_size = get_dataset_only_chord_1(beats_raw)
    if choice == 2:
        beats, vocab_sizes, target_size = get_dataset_only_chord_2(beats_raw)
    if choice == 4:
        beats, vocab_sizes, target_size = get_dataset4(melody_raw, beats_raw)
    if choice == 5 or choice == 6 or choice == 7 or choice== 8:
        beats, vocab_sizes, target_size = get_dataset5(melody_raw, beats_raw)

    sequences = []          # store chord as multi-hot
    target_sequence = []    # store chord as one-hot
    num_mels = beats['melid'].max()
    melodies = []
    bass_pitch = []

    # for each song load its chord seq
    for i in range(1, num_mels+1):
        song = beats.loc[beats['melid'] == i]
        seq_pitch = song['Root_pitch_num'].to_numpy()
        seq_one_hot = song['new_chord_num'].to_numpy()

        if choice == 2:
            seq_chord_info = song['chord_info_num'].to_numpy()
        else:
            seq_triad = song['triad_num'].to_numpy()
            seq_added_note = song['added_note_num'].to_numpy()

        if len(seq_pitch) > 1:
            # One input for each melody note
            if choice == 3: 
                seq_pitch_mel = song['pitch_encoded'].to_numpy()
                sequences.append(np.array([seq_pitch, seq_chord_info, seq_pitch_mel]).T)

            # Melody embedding includes all notes played during chord
            else:  
                if choice == 2:
                    sequences.append(np.array([seq_pitch, seq_chord_info]).T)
                else:
                    sequences.append(np.array([seq_pitch, seq_triad, seq_added_note]).T)
                    if choice == 4 or choice == 8:     # melody encoding
                        melody_encoding = song['pitch_sequence']
                        melodies.append(np.array(melody_encoding))
                    if choice == 5:     # bass encoding
                        bass_pitch_encoding = song['bass_pitch_sequence']
                        bass_pitch.append(np.array(bass_pitch_encoding))
                    if choice == 6:     # melody + bass encoding
                        melody_encoding = song['pitch_sequence']
                        bass_pitch_encoding = song['bass_pitch_sequence']
                        melodies.append(np.array(melody_encoding))
                        bass_pitch.append(np.array(bass_pitch_encoding))
                    if choice == 7:     # melody weighted with the duration of the notes
                        duration_melody = song['duration_sequence']
                        melody_encoding = song['pitch_sequence']
                        # Append a tuple of the melody and duration
                        melodies.append(np.array([(melody_encoding, duration_melody)], dtype='object, object'))


            target_sequence.append(seq_one_hot)
                
    # convert to np array
    sequences = np.array(sequences, dtype=object)
    target_sequence = np.array(target_sequence, dtype=object)
    melodies = np.array(melodies, dtype=object)
    bass_pitch = np.array(bass_pitch, dtype=object)

    if choice == 4:
        dataset = MultiHot_MelodyEncoded_VLDataset(sequences, melodies, target_sequence, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return dataset, input_size, target_size

    if choice == 5:
        dataset = MultiHot_MelodyEncoded_VLDataset(sequences, bass_pitch, target_sequence, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return dataset, input_size, target_size

    if choice == 6:
        dataset = MultiHot_MelodyBassEncoded_VLDataset(sequences, melodies, bass_pitch, target_sequence, vocab_sizes)
        input_size = sum(vocab_sizes) + 12 + 12
        return dataset, input_size, target_size

    if choice == 7:
        dataset = MultiHot_MelodyDurationEncoded_VLDataset(sequences, melodies, target_sequence, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return dataset, input_size, target_size

    if choice == 8:    
        dataset = MultiHot_MelodyWeighted_VLDataset(sequences, melodies, target_sequence, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return dataset, input_size, target_size

    dataset = MultiHot_VLDataset(sequences, target_sequence, vocab_sizes)

    input_size = sum(vocab_sizes)
    return dataset, input_size, target_size

# if __name__ == "__main__":
#     get_dataset_multi_hot(choice=1)