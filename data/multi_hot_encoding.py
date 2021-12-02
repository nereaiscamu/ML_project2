from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from dataset import OneHot_VLDataset, MultiHot_VLDataset, MultiHot_MelodyEncoded_VLDataset, MultiHot_MelodyBassEncoded_VLDataset
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

def encode_chords(beats):
    """
    Encode the chords:
        (1): Root pitch and '#'. Vocab size = 13
        (2): Wether or not it is minor
        (3): Other chord info
    """
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

    ''' Now only 12 pitches + No Chord, map it to the MIDI numeric notation for each pitch'''

    # Process chord info other than root note
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

    # Save new chord 
    beats['new_chord'] = beats['final_pitch'].str.cat(beats['chord_info'])

    # Separate minor from chord_info
    beats['minor'] = 0
    beats.loc[beats['chord_info'].str.contains('-') == True, 'minor' ] = 1
    beats['chord_info'] = beats['chord_info'].str.replace('-','')

    return beats


def get_dataset2(beats):
    '''
    Dataset 2: Multi-hot. Combination of 3 One-Hot vectors:
        (1): Root pitch and '#'. Vocab size = 13
        (2): 
        (3):
        
    '''
    beats['chord'].replace('', np.nan, inplace=True)
    beats = beats[['beatid', 'melid', 'chord']].dropna()

    beats = encode_chords(beats)

    unique_chords = pd.unique(beats['new_chord'])
    unique_pitch = pd.unique(beats['final_pitch'])
    unique_minor = pd.unique(beats['minor'])
    unique_chord_info = pd.unique(beats['chord_info'])
    #vocab_sizes = [len(unique_pitch), 1, len(unique_chord_info)]   # FIXME should use a flag instead of a 2-dim one-hot?
    vocab_sizes = [len(unique_pitch), len(unique_minor), len(unique_chord_info)]
    
    # Encode each final pitch to a number by order of appearnce
    final_pitch_map = {}
    for i, c in enumerate(unique_pitch):
        final_pitch_map[c] = i
    beats['final_pitch_num'] = beats['final_pitch'].map(final_pitch_map)

    # Encode chord info
    chord_info_map = {}
    for i, c in enumerate(unique_chord_info):
        chord_info_map[c] = i
    beats['chord_info_num'] = beats['chord_info'].map(chord_info_map)

    # Encode new chord
    new_chord_map = {}
    for i, c in enumerate(unique_chords):
        new_chord_map[c] = i
    beats['new_chord_num'] = beats['new_chord'].map(new_chord_map)

    target_size = len(unique_chords)
    print('\nUsing a total chord vocab size of %d (One-hot)' % len(unique_chords))
    print('Multi-hot input:')
    print('\t(1): Root pitch and #. Vocab size of %d' % len(unique_pitch))
    print('\t(2): Whether or not it is minor. Binary flag, size %d' % len(unique_minor))
    print('\t(3): Other chord info. Vocab size of %d\n' % len(unique_chord_info))

    return beats, vocab_sizes, target_size


def get_dataset3(melody, beats):
    """
    Dataset 3: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Wether or not it is minor
        (3): Other chord info
        (4): Melody pitch. One entry for each note played
    """
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    beats_encoded = encode_chords(beats)
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
        (2): Wether or not it is minor
        (3): Other chord info
        (4): Melody encoding. All notes combined into one embedding
    """
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    beats_encoded = encode_chords(beats)
    mel_beats = encode_pitch(melody, beats_encoded, pitch_sequence=True)

    unique_chords = pd.unique(mel_beats['new_chord'])
    unique_pitch = pd.unique(mel_beats['final_pitch'])
    unique_minor = pd.unique(mel_beats['minor'])
    unique_chord_info = pd.unique(mel_beats['chord_info'])
    unique_notes = pd.unique(mel_beats['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_minor), len(unique_chord_info)]
    
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

def get_dataset5(melody, beats):
    """
    Dataset 4: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Wether or not it is minor
        (3): Other chord info
        (4): Melody encoding. All notes combined into one embedding
        (5): Bass pitch encoding
    """
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    beats_encoded = encode_chords(beats)
    mel_beats = encode_pitch(melody, beats_encoded, pitch_sequence=True)

    unique_chords = pd.unique(mel_beats['new_chord'])
    unique_pitch = pd.unique(mel_beats['final_pitch'])
    unique_minor = pd.unique(mel_beats['minor'])
    unique_chord_info = pd.unique(mel_beats['chord_info'])
    unique_notes = pd.unique(mel_beats['pitch_encoded'])
    vocab_sizes = [len(unique_pitch), len(unique_minor), len(unique_chord_info)]
    
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
    print('\t(4): Melody pitch. Vocab size of %d' % len(unique_notes))
    print('\t(5): Bass pitch. Vocab size of %d\n' % len(unique_notes))

    return mel_beats, vocab_sizes, target_size

def get_dataset_multi_hot(choice=2, test_split=0.2):
    '''
    Generate train and test dataset. Based on dataset choice
    choice:
        2: Multi-hot chord encoding
        3: Multi-hot chord encoding + melody encoding (event for every note)
        4: Multi-hot chord encoding + melody encoding (notes encoding combined)
        5: Multi-hot chord encoding + bass pitch encoding
        6: Multi-hot chord encoding + melody encoding & bass pitch encoding
    '''

    path = "./Data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
    engine = create_engine(f"sqlite:///{path}")
    beats_raw = pd.read_sql("beats", engine)
    melody_raw = pd.read_sql("melody", engine)

    if choice == 2:
        beats, vocab_sizes, target_size = get_dataset2(beats_raw)
    if choice == 3:
        beats, vocab_sizes, target_size = get_dataset3(melody_raw, beats_raw)
    if choice == 4:
        beats, vocab_sizes, target_size = get_dataset4(melody_raw, beats_raw)
    if choice == 5 or choice == 6:
        beats, vocab_sizes, target_size = get_dataset5(melody_raw, beats_raw)

    sequences = []          # store chord as multi-hot
    target_sequence = []   # store chord as one-hot
    num_mels = beats['melid'].max()
    melodies = []
    bass_pitch = []
    # for each song load its chord seq
    for i in range(1, num_mels+1):
        song = beats.loc[beats['melid'] == i]
        seq_pitch = song['final_pitch_num'].to_numpy()

        seq_minor = song['minor'].to_numpy()
        seq_chord_info = song['chord_info_num'].to_numpy()
        seq_one_hot = song['new_chord_num'].to_numpy()
        if choice == 2:    
            if len(seq_pitch) > 1:
                sequences.append(np.array([seq_pitch, seq_minor, seq_chord_info]).T)
                target_sequence.append(seq_one_hot)
        if choice == 3:
            seq_pitch_mel = song['pitch_encoded'].to_numpy()
            if len(seq_pitch) > 1:
                sequences.append(np.array([seq_pitch, seq_minor, seq_chord_info, seq_pitch_mel]).T)
                target_sequence.append(seq_one_hot)
        if choice == 4:
            melody_encoding = song['pitch_sequence']
            if len(seq_pitch) > 1:
                sequences.append(np.array([seq_pitch, seq_minor, seq_chord_info]).T)
                melodies.append(np.array(melody_encoding))
                target_sequence.append(seq_one_hot)
        if choice == 5:
            bass_pitch_encoding = song['bass_pitch_sequence']
            if len(seq_pitch) > 1:
                sequences.append(np.array([seq_pitch, seq_minor, seq_chord_info]).T)
                bass_pitch.append(np.array(bass_pitch_encoding))
                target_sequence.append(seq_one_hot)
        if choice == 6:
            melody_encoding = song['pitch_sequence']
            bass_pitch_encoding = song['bass_pitch_sequence']
            if len(seq_pitch) > 1:
                sequences.append(np.array([seq_pitch, seq_minor, seq_chord_info]).T)
                melodies.append(np.array(melody_encoding))
                bass_pitch.append(np.array(bass_pitch_encoding))
                target_sequence.append(seq_one_hot)
                

    # Split
    split_idx = int(len(sequences)*(1-test_split))
    train_seq = sequences[:split_idx]
    test_seq = sequences[split_idx:]
    train_target_seq = target_sequence[:split_idx]
    test_target_seq = target_sequence[split_idx:]

    if choice == 4:
        train_mel = melodies[:split_idx]
        test_mel = melodies[split_idx:]
        train_dataset = MultiHot_MelodyEncoded_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyEncoded_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return train_dataset, test_dataset, input_size, target_size

    if choice == 5:
        train_bass = bass_pitch[:split_idx]
        test_bass = bass_pitch[split_idx:]
        train_dataset = MultiHot_MelodyEncoded_VLDataset(train_seq, train_bass, train_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyEncoded_VLDataset(test_seq, test_bass, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
        return train_dataset, test_dataset, input_size, target_size

    if choice == 6:
        train_mel = melodies[:split_idx]
        test_mel = melodies[split_idx:]
        train_bass = bass_pitch[:split_idx]
        test_bass = bass_pitch[split_idx:]        
        train_dataset = MultiHot_MelodyBassEncoded_VLDataset(train_seq, train_mel, train_bass, train_target_seq, vocab_sizes)
        test_dataset = MultiHot_MelodyBassEncoded_VLDataset(test_seq, test_mel, test_bass, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12 + 12
        return train_dataset, test_dataset, input_size, target_size

    train_dataset = MultiHot_VLDataset(train_seq, train_target_seq, vocab_sizes)
    test_dataset = MultiHot_VLDataset(test_seq, test_target_seq, vocab_sizes)

    input_size = sum(vocab_sizes)
    return train_dataset, test_dataset, input_size, target_size


if __name__ == "__main__":
    get_dataset_multi_hot(choice=3)
    
