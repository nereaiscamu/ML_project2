from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from dataset import MultiHot_VLDataset, MultiHot_MelodyEncoded_VLDataset, MultiHot_MelodyBassEncoded_VLDataset, MultiHot_MelodyDurationEncoded_VLDataset, MultiHot_MelodyWeighted_VLDataset
import pickle
import sys
from data.combine_melody_beats import encode_melody

sys.path.append('../')

path = "./data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
engine = create_engine(f"sqlite:///{path}")
beats = pd.read_sql("beats", engine)
melody= pd.read_sql("melody", engine)


# *************** CHORD ENCODING *******************

def preprocess_chords(beats, mel_included = False):
    # Remove empty rows, if melody encoded, empty rows corresponds to -1
    if mel_included == False:
        beats= beats.loc[beats['chord'] != 'NC']
        beats['chord'].replace('', np.nan, inplace=True)
        beats= beats.loc[~beats['chord'].isna()]
        
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

    beats = beats.drop(['mod', 'mod2', 'complete_pitch', 'root_pitch'], axis=1)

    # NOW WE ADD A SECOND VECTOR WITH ALL THE INFO ABOUT THE CHORD BUT THE ROOT PITCH

    ''' Remove or change useless information'''

    beats['chord_map'] = beats['chord_info']
    beats['chord_map'] = beats['chord_map'].str.replace('\/(.*)','', regex=True)
    beats['chord_map'] = beats['chord_map'].str.replace('m','-', regex=True)
    beats['chord_map'] = beats['chord_map'].str.replace('7alt','7', regex=True)
    
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
        beats['added_note'] = beats['added_note'].str.replace(i,'', regex=True)
    
    # in the added note, diminished is only kept when it affects the 7th note
    # if only for the triad, already in the previous vector
    beats.loc[ beats['added_note'] == 'o', "added_note"] = ''
    beats.loc[ beats['added_note'] == '7', "added_note"] = 'm7'
    beats.loc[beats['added_note'] == '', 'added_note'] = 'none'
        
    # 'triad' contains 6 elements. 'added_note' contains 6 elements. Option 2: Root pitch (13) + triad (6) + extra_node (6)

    return beats

def encode_chords(table):
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


# *************** DATASETS ONLY CHORD ENCODING *******************

def get_dataset_baseline(beats):
    '''
    Dataset Baseline: Multi-hot. Combination of 3 One-Hot vectors:
        (1): Root pitch and '#'. Vocab size = 12
        (2): triad. Vocab size = 6
        (3): Extra note. Vocab size = 5
    Total: 156 unique chords

    Args:
        beats: Beats table
    Returns:
        Encoded beats table, the vocabulary size and the target size
    '''
    beats = beats[['beatid', 'melid', 'chord', 'bar', 'beat', 'bass_pitch']]
    beats = preprocess_chords(beats, mel_included = False)
    beats['new_chord'] = beats['new_chord'].loc[beats['new_chord'].shift(-1) != beats['new_chord']]
    beats= beats.loc[~beats['new_chord'].isna()]
    beats = encode_chords(beats)

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


# *************** DATASET CHORD + MELODY/BASS ENCODING *******************

def get_dataset_melody(melody, beats):
    """
    Dataset Melody: Multi-hot. Combination of 3 One-Hot vectors for chord encoding + Melody and Bass encoding
        (1): Root pitch and '#'. Vocab size = 13
        (2): Whether or not it is minor
        (3): Other chord info
        (4): Melody encoding. All notes combined into one embedding
        (5): Bass pitch encoding

    Args:
        melody: Melody table
        beats: Beats table
    Returns:
        Combined and encoded table of melody and beats table, the vocabulary size and the target size
    """
    
    beats = beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]


    beats = preprocess_chords(beats, mel_included = True)
    beats = encode_chords(beats)
    beats_mel = encode_melody(melody, beats, pitch_sequence=True)    

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
    print('\t(4): Melody. Vocab size of %d' % len(unique_notes))
    print('\t(5): Bass pitch. Vocab size of %d\n' % len(unique_notes))

    return beats_mel, vocab_sizes, target_size


# *************** DATASETS -- ALL COMBINATIONS *******************

def get_dataset_multi_hot(choice=1, val_split=0.1, test_split=0.1, seed=42, get_tune_ids=False, return_seq_melid=0, kfold=False):
    '''
    Generate train and test dataset. Based on dataset choice
    choice:
        (1)Baseline        : Multi-hot chord encoding (12+6+5)
        (2)Melody          : Multi-hot chord encoding (12+6+5) + melody encoding (notes encoding combined)
        (3)Bass            : Multi-hot chord encoding (12+6+5) + bass pitch encoding
        (4)Melody + Bass   : Multi-hot chord encoding (12+6+5) + melody encoding & bass pitch encoding
        (5)Melody duration : Multi-hot chord encoding (12+6+5) + melody encoding with weighted duration
        (6)Melody weighted : Multi-hot chord encoding (12+6+5) + weighted melody encoding

    Args:
        choice: Choice of dataset depending on model. Between 1 and 6
        val_split: Size of the validation set. Between 0 and 1
        test_split: Size of the test set. Between 0 and 1
        seed: Random seed
        get_tune_ids: If true, returns also a list with the corresponding melid of each tune
        return_seq_melid: If > 0, returns the sequence of melody for the given tune
        kfold: If true, returns the complete dataset (not splitted) for the cross-validation
    Returns:
        Dataset either split into train, validation and test or complete dataset and the 
        Input and Target sizes
    '''

    path = "./data/wjazzd.db" # REPLACE THIS WITH PATH TO FILE
    engine = create_engine(f"sqlite:///{path}")
    beats_raw = pd.read_sql("beats", engine)
    melody_raw = pd.read_sql("melody", engine)

    if choice == 1:
        beats, vocab_sizes, target_size = get_dataset_baseline(beats_raw)
    else:
        beats, vocab_sizes, target_size = get_dataset_melody(melody_raw, beats_raw)

    sequences = []          # store chord as multi-hot
    target_sequence = []    # store chord as one-hot
    num_mels = beats['melid'].max()
    melodies = []
    bass_pitch = []
    song_ids = []

    # for each song load its chord seq
    for i in range(1, num_mels+1):
        song = beats.loc[beats['melid'] == i]
        seq_pitch = song['Root_pitch_num'].to_numpy()
        seq_one_hot = song['new_chord_num'].to_numpy()
        seq_triad = song['triad_num'].to_numpy()
        seq_added_note = song['added_note_num'].to_numpy()

        if return_seq_melid > 0:
            if choice!=1:
                if i == return_seq_melid:
                    mel_sequence = song['pitch_sequence']
                    return mel_sequence
            else:
                print('\nChoice 1 does not contain the melody encoding!\n')

        if len(seq_pitch) > 1:
            song_ids.append(i)

            # Melody embedding includes all notes played during chord
            sequences.append(np.array([seq_pitch, seq_triad, seq_added_note]).T)            
            if choice != 1:
                bass_pitch_encoding = song['bass_pitch_sequence']
                bass_pitch.append(np.array(bass_pitch_encoding))
                melody_encoding = song['pitch_sequence']
                if choice == 5: 
                    # melody weighted with the duration of the notes    
                    duration_melody = song['duration_sequence']
                    # Append a tuple of the melody and duration
                    melodies.append(np.array([(melody_encoding, duration_melody)], dtype='object, object'))
                else:
                    melodies.append(np.array(melody_encoding))

            target_sequence.append(seq_one_hot)
                
    # convert to np array
    sequences = np.array(sequences, dtype=object)
    target_sequence = np.array(target_sequence, dtype=object)
    melodies = np.array(melodies, dtype=object)
    bass_pitch = np.array(bass_pitch, dtype=object)

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

    if choice != 1:
        train_mel = melodies[train_idxs]
        val_mel = melodies[val_idxs]
        test_mel = melodies[test_idxs]
        train_bass = bass_pitch[train_idxs]
        val_bass = bass_pitch[val_idxs]
        test_bass = bass_pitch[test_idxs]


    if choice == 1:
        if kfold:
            dataset = MultiHot_VLDataset(sequences, target_sequence, vocab_sizes)
        else:
            train_dataset = MultiHot_VLDataset(train_seq, train_target_seq, vocab_sizes)
            val_dataset = MultiHot_VLDataset(val_seq, val_target_seq, vocab_sizes)
            test_dataset = MultiHot_VLDataset(test_seq, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes)
    elif choice == 2:
        if kfold:
            dataset = MultiHot_MelodyEncoded_VLDataset(sequences, melodies, target_sequence, vocab_sizes)
        else:
            train_dataset = MultiHot_MelodyEncoded_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
            val_dataset = MultiHot_MelodyEncoded_VLDataset(val_seq, val_mel, val_target_seq, vocab_sizes)
            test_dataset = MultiHot_MelodyEncoded_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
    elif choice == 3:
        if kfold:
            dataset = MultiHot_MelodyEncoded_VLDataset(sequences, bass_pitch, target_sequence, vocab_sizes)
        else:
            train_dataset = MultiHot_MelodyEncoded_VLDataset(train_seq, train_bass, train_target_seq, vocab_sizes)
            val_dataset = MultiHot_MelodyEncoded_VLDataset(val_seq, val_bass, val_target_seq, vocab_sizes)
            test_dataset = MultiHot_MelodyEncoded_VLDataset(test_seq, test_bass, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
    elif choice == 4:
        if kfold:
            dataset = MultiHot_MelodyBassEncoded_VLDataset(sequences, melodies, bass_pitch, target_sequence, vocab_sizes)
        else:
            train_dataset = MultiHot_MelodyBassEncoded_VLDataset(train_seq, train_mel, train_bass, train_target_seq, vocab_sizes)
            val_dataset = MultiHot_MelodyBassEncoded_VLDataset(val_seq, val_mel, val_bass, val_target_seq, vocab_sizes)
            test_dataset = MultiHot_MelodyBassEncoded_VLDataset(test_seq, test_mel, test_bass, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12 + 12
    elif choice == 5: 
        if kfold:
            dataset = MultiHot_MelodyDurationEncoded_VLDataset(sequences, melodies, target_sequence, vocab_sizes)
        else:   
            train_dataset = MultiHot_MelodyDurationEncoded_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
            val_dataset = MultiHot_MelodyDurationEncoded_VLDataset(val_seq, val_mel, val_target_seq, vocab_sizes)
            test_dataset = MultiHot_MelodyDurationEncoded_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12
    elif choice == 6: 
        if kfold:
            dataset = MultiHot_MelodyWeighted_VLDataset(sequences, melodies, target_sequence, vocab_sizes)
        else:      
            train_dataset = MultiHot_MelodyWeighted_VLDataset(train_seq, train_mel, train_target_seq, vocab_sizes)
            val_dataset = MultiHot_MelodyWeighted_VLDataset(val_seq, val_mel, val_target_seq, vocab_sizes)
            test_dataset = MultiHot_MelodyWeighted_VLDataset(test_seq, test_mel, test_target_seq, vocab_sizes)
        input_size = sum(vocab_sizes) + 12

    # save dataset
    if kfold:
        data = (dataset, input_size, target_size)
        with open('data/datasets/dataset_wo_split' + str(choice) + '.pickle', 'wb') as f:
            pickle.dump(data, f)
    else:
        data = (train_dataset, val_dataset, test_dataset, input_size, target_size)
        with open('data/datasets/dataset' + str(choice) + '.pickle', 'wb') as f:
            pickle.dump(data, f)

    if get_tune_ids:
        return train_dataset, val_dataset, test_dataset, input_size, target_size, song_ids
    if kfold:
        return dataset, input_size, target_size
    else:
        return train_dataset, val_dataset, test_dataset, input_size, target_size
