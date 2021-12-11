from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from dataset import OneHot_VLDataset, MultiHot_VLDataset, MultiHot_MelodyEncoded_VLDataset, MultiHot_MelodyBassEncoded_VLDataset, MultiHot_MelodyDurationEncoded_VLDataset, MultiHot_MelodyWeighted_VLDataset
import pdb

import sys
sys.path.append('../')
from data.combine_melody_beats import encode_pitch


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

def preprocess_chords(beats):
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

    beats['final_pitch'] = beats['complete_pitch']
    beats.loc[ beats['complete_pitch'] == 'Db', "final_pitch"] = 'C#'
    beats.loc[ beats['complete_pitch'] == 'Eb', "final_pitch"] = 'D#'
    beats.loc[ beats['complete_pitch'] == 'Fb', "final_pitch"] = 'E'
    beats.loc[ beats['complete_pitch'] == 'Gb', "final_pitch"] = 'F#'
    beats.loc[ beats['complete_pitch'] == 'Ab', "final_pitch"] = 'G#'
    beats.loc[ beats['complete_pitch'] == 'Bb', "final_pitch"] = 'A#'
    beats.loc[ beats['complete_pitch'] == 'Cb', "final_pitch"] = 'B'

    # *** Now only 12 pitches + No Chord ***

    beats = beats[['beatid', 'melid', 'chord', 'final_pitch', 'chord_info', 'bar', 'beat', 'bass_pitch']] #remove useless columns

    beats['chord_info'] = beats['chord_info'].str.replace('\/(.*)','')
    beats['chord_info'] = beats['chord_info'].str.replace('j','')
    beats['chord_info'] = beats['chord_info'].str.replace('m','-')
    chord_info_importance = beats['chord_info'].value_counts()

    info_to_remove = ['9#','9b', '9', '11b', '11#', '11', '13b', '13#', '13', 'C']
    for j in info_to_remove: 
        beats['chord_info'] = beats['chord_info'].str.replace(j,'')
    
    # Save new chord 
    beats['new_chord'] = beats['final_pitch'].str.cat(beats['chord_info'])

    # 'chord_info' contains 14 elements. Option 1: Root pitch (13) + Chord_info (14)
    chord_info_all = pd.unique(beats['chord_info'])

    modes = ['o', '-', '+', 'sus', 'alt']
    beats['mode'] = '0'
    beats['extra_note'] = beats['chord_info']

    for i in modes:
        beats.loc[beats['chord_info'].str.contains(i, regex = False) == True, 'mode'] = i
        beats['extra_note'] = beats['extra_note'].str.replace(i,'', regex = False)
    
    # 'mode' contains 6 elements. 'extra_note' contains 4 elements. Option 2: Root pitch (13) + mode (6) + extra_node (4)

    return beats

def encode_chords_1(table):
    unique_chords = pd.unique(table['new_chord'])
    unique_pitch = pd.unique(table['final_pitch'])
    unique_mode = pd.unique(table['mode'])
    unique_extra_note = pd.unique(table['extra_note'])
    
    # Encode each final pitch to a number by order of appearnce
    final_pitch_map = {}
    for i, c in enumerate(unique_pitch):
        final_pitch_map[c] = i
    table['final_pitch_num'] = table['final_pitch'].map(final_pitch_map)

    # Encode extra note
    mode_map = {}
    for i, c in enumerate(unique_mode):
        mode_map[c] = i
    table['mode_num'] = table['mode'].map(mode_map)

    # Encode extra note
    extra_note_map = {}
    for i, c in enumerate(unique_extra_note):
        extra_note_map[c] = i
    table['extra_note_num'] = table['extra_note'].map(extra_note_map)

    # Encode new chord
    new_chord_map = {}
    for i, c in enumerate(unique_chords):
        new_chord_map[c] = i
    table['new_chord_num'] = table['new_chord'].map(new_chord_map)

    return table

def encode_chords_2(table):
    unique_chords = pd.unique(table['new_chord'])
    unique_pitch = pd.unique(table['final_pitch'])
    unique_chord_info = pd.unique(table['chord_info'])
    vocab_sizes = [len(unique_pitch), len(unique_chord_info)]
    
    # Encode each final pitch to a number by order of appearnce
    final_pitch_map = {}
    for i, c in enumerate(unique_pitch):
        final_pitch_map[c] = i
    table['final_pitch_num'] = table['final_pitch'].map(final_pitch_map)

    # Encode chord map
    chord_info_map = {}
    for i, c in enumerate(unique_chord_info):
        chord_info_map[c] = i
    table['chord_info_num'] = table['chord_info'].map(chord_info_map)

    # Encode new chord
    new_chord_map = {}
    for i, c in enumerate(unique_chords):
        new_chord_map[c] = i
    table['new_chord_num'] = table['new_chord'].map(new_chord_map)

    return table

def get_dataset_only_chord_1(beats):
    '''
    Dataset 2: Multi-hot. Combination of 3 One-Hot vectors:
        (1): Root pitch and '#'. Vocab size = 13
        (2): Mode. Vocab size = 6
        (3): Extra note. Vocab size = 4
    '''
    beats['chord'].replace('', np.nan, inplace=True)
    beats = beats[['beatid', 'melid', 'chord', 'bar', 'beat', 'bass_pitch']].dropna()
    beats = preprocess_chords(beats)
    beats = encode_chords_1(beats)

    unique_chords = pd.unique(beats['new_chord'])
    unique_pitch = pd.unique(beats['final_pitch'])
    unique_mode = pd.unique(beats['mode'])
    unique_extra_note = pd.unique(beats['extra_note'])
    vocab_sizes = [len(unique_pitch), len(unique_mode), len(unique_extra_note)]
    target_size = len(unique_chords)

    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Mode. Vocab size of %d' % len(unique_mode))
    print('\t(3): Extra note. Vocab size of %d\n' % len(unique_extra_note))

    return beats, vocab_sizes, target_size

def get_dataset_only_chord_2(beats):
    '''
    Dataset 2: Multi-hot. Combination of 3 One-Hot vectors:
        (1): Root pitch and '#'. Vocab size = 13
        (2): Chord info. Vocab size = 14
    '''
    beats['chord'].replace('', np.nan, inplace=True)
    beats = beats[['beatid', 'melid', 'chord']].dropna()
    beats = preprocess_chords(beats)

    beats = encode_chords_2(beats)

    unique_chords = pd.unique(beats['new_chord'])
    unique_pitch = pd.unique(beats['final_pitch'])
    unique_chord_info = pd.unique(beats['chord_info'])
    vocab_sizes = [len(unique_pitch), len(unique_chord_info)]
    target_size = len(unique_chords)

    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Chord info. Vocab size of %d\n' % len(unique_chord_info))

    return beats, vocab_sizes, target_size


def get_dataset3(melody, beats):
    """
    Dataset 3: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Whether or not it is minor
        (3): Other chord info
        (4): Melody pitch. One entry for each note played
    """
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    beats_encoded = preprocess_chords(beats)
    mel_beats = encode_pitch(melody, beats_encoded, pitch_sequence=False)

    unique_chords = pd.unique(mel_beats['new_chord'])
    unique_pitch = pd.unique(mel_beats['final_pitch'])
    unique_minor = pd.unique(mel_beats['minor'])
    unique_chord_info = pd.unique(mel_beats['chord_info'])
    unique_notes = pd.unique(mel_beats['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_minor), len(unique_chord_info), len(unique_notes)]
    
    # Encode each final pitch to a number by order of appearnce
    final_pitch_map = {}
    for i, c in enumerate(unique_pitch):
        final_pitch_map[c] = i
    mel_beats['final_pitch_num'] = mel_beats['final_pitch'].map(final_pitch_map)

    # Encode chord info
    chord_info_map = {}
    for i, c in enumerate(unique_chord_info):
        chord_info_map[c] = i
    mel_beats['chord_info_num'] = mel_beats['chord_info'].map(chord_info_map)

    # Encode new chord
    new_chord_map = {}
    for i, c in enumerate(unique_chords):
        new_chord_map[c] = i
    mel_beats['new_chord_num'] = mel_beats['new_chord'].map(new_chord_map)

    target_size = len(unique_chords)
    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Whether or not it is minor. Binary flag, size %d' % len(unique_minor))
    print('\t(3): Other chord info. Vocab size of %d' % len(unique_chord_info))
    print('\t(4): Melody pitch. Vocab size of %d\n' % len(unique_notes))

    return mel_beats, vocab_sizes, target_size

def get_dataset4(melody, beats):
    """
    Dataset 4: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Whether or not it is minor
        (3): Other chord info
        (4): Melody encoding. All notes combined into one embedding
    """
    beats['chord'].replace('', np.nan, inplace=True)
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']].dropna()

    beats = preprocess_chords(beats)
    beats = encode_chords_1(beats)

    mel_beats = encode_pitch(melody, beats, pitch_sequence=True)

    unique_chords = pd.unique(mel_beats['new_chord'])
    unique_pitch = pd.unique(mel_beats['final_pitch'])
    unique_mode = pd.unique(beats['mode'])
    unique_extra_note = pd.unique(beats['extra_note'])
    unique_notes = pd.unique(mel_beats['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_mode), len(unique_extra_note)]
    target_size = len(unique_chords)

    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Mode. Vocab size of %d' % len(unique_mode))
    print('\t(3): Extra note. Vocab size of %d' % len(unique_extra_note))
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

    beats_encoded = preprocess_chords(beats)
    beats_mel = encode_pitch(melody, beats_encoded, pitch_sequence=True)
    beats_mel = encode_chords_1(beats_mel)

    unique_chords = pd.unique(beats_mel['new_chord'])
    unique_pitch = pd.unique(beats_mel['final_pitch'])
    unique_mode = pd.unique(beats_mel['mode'])
    unique_extra_note = pd.unique(beats_mel['extra_note'])
    unique_notes = pd.unique(beats_mel['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_mode), len(unique_extra_note)]
    target_size = len(unique_chords)
    
    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Mode. Vocab size of %d' % len(unique_mode))
    print('\t(3): Extra note. Vocab size of %d' % len(unique_extra_note))
    print('\t(4): Melody pitch. Vocab size of %d\n' % len(unique_notes))

    return beats_mel, vocab_sizes, target_size

def get_dataset_multi_hot(choice=1, val_split=0.1, test_split=0.1):
    '''
    Generate train and test dataset. Based on dataset choice
    choice:
        2: Multi-hot chord encoding
        3: Multi-hot chord encoding + melody encoding (event for every note)
        4: Multi-hot chord encoding + melody encoding (notes encoding combined)
        5: Multi-hot chord encoding + bass pitch encoding
        6: Multi-hot chord encoding + melody encoding & bass pitch encoding
        7: Multi-hot chord encoding + melody encoding with weighted duration
    '''

    path = "./Data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
    engine = create_engine(f"sqlite:///{path}")
    beats_raw = pd.read_sql("beats", engine)
    melody_raw = pd.read_sql("melody", engine)

    if choice == 1:
        beats, vocab_sizes, target_size = get_dataset_only_chord_1(beats_raw)
    if choice == 2:
        beats, vocab_sizes, target_size = get_dataset_only_chord_2(beats_raw)
    if choice == 3:
        beats, vocab_sizes, target_size = get_dataset3(melody_raw, beats_raw)
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
        seq_pitch = song['final_pitch_num'].to_numpy()
        seq_one_hot = song['new_chord_num'].to_numpy()

        if choice == 2:
            seq_chord_info = song['chord_info_num'].to_numpy()
        else:
            seq_mode = song['mode_num'].to_numpy()
            seq_extra_note = song['extra_note_num'].to_numpy()

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
                    sequences.append(np.array([seq_pitch, seq_mode, seq_extra_note]).T)
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
    sequences = np.array(sequences)
    target_sequence = np.array(target_sequence)
    melodies = np.array(melodies)
    bass_pitch = np.array(bass_pitch)

    # Split Train/Val/Test
    random_idxs = np.random.permutation(len(sequences))
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
    return train_dataset, val_dataset, test_dataset, input_size, target_size


if __name__ == "__main__":
    get_dataset_multi_hot(choice=3)
    
