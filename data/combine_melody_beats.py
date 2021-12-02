import numpy as np
import pandas as pd
import pdb

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

    df_beats_mel = concat_melody_beats(df_melody, df_beats)
    
    # Encode pitch
    df_beats_mel['pitch_encoded'] = np.mod(df_beats_mel['pitch'], 12)
    df_beats_mel['bass_pitch_encoded'] = np.mod(df_beats_mel['bass_pitch'], 12)

    df_beats_mel.fillna(-1, inplace=True)
    df_beats_mel['pitch_encoded'] = df_beats_mel['pitch_encoded'].astype(int)
    max_pitch = df_beats_mel['pitch_encoded'].max()
    df_beats_mel['bass_pitch_encoded'] = df_beats_mel['bass_pitch_encoded'].astype(int)

    ## Encode pitch for every chord of melody
    if not pitch_sequence:
        return df_beats_mel

    ## Encode sequence of pitch for every chord
    # Add column that represent chord changes
    df_beats_mel['chord_changed'] = (df_beats_mel['chord'].shift() != df_beats_mel["chord"]).cumsum()

    # Group chord changes to get sequences
    pitch_sequences = [g['pitch_encoded'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]
    bass_pitch_sequences = [g['bass_pitch_encoded'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]

    # Identify last row of current chord
    df_beats_mel['pitch_sequence'] = (df_beats_mel['chord'].shift(-1) != df_beats_mel["chord"])

    # Change type to type object to add list to cell
    df_beats_mel['pitch_sequence'] = df_beats_mel['pitch_sequence'].astype(object)
    df_beats_mel['bass_pitch_sequence'] = df_beats_mel['pitch_sequence']

    # Set sequence to last chord
    # TODO decrease running time if possible
    # TODO discuss where the sequence should be added
    for idx, _ in df_beats_mel.iterrows():
        if df_beats_mel.at[idx, 'pitch_sequence'] == True:
            df_beats_mel.at[idx, 'pitch_sequence'] = pitch_sequences.pop(0)
            df_beats_mel.at[idx, 'bass_pitch_sequence'] = bass_pitch_sequences.pop(0)

    # melody_encoded = [np.zeros(max_pitch) for _ in range(len(df_beats_mel))]
    # df_beats_mel['melody_encoded'] = melody_encoded
    df_beats_mel.drop(['bass_pitch_encoded', 'chord_changed'], axis=1, inplace=True)
    df_beats_mel.drop(df_beats_mel[df_beats_mel['pitch_sequence'] == False].index, inplace=True)

    return df_beats_mel


def concat_melody_beats(df_melody, df_beats):
    """
    Fills the chords in the beats table and combines the melody table and the beats table 
    using the MultiIndex key of (melid, bar, beat).

    Args:
        df_melody: Dataframe of the melody table
        df_beats: Dataframe of the beats table

    Returns:
        Dataframe consisting of the combination of both tables
    """
    # Remove useless columns
    df_melody = df_melody[['eventid', 'melid', 'pitch', 'duration', 'bar', 'beat']]
    #df_beats = df_beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    # Replace empty strings by nan and then use ffill to fill with last seen chord
    #df_chords = df_beats.replace({'chord': {'': np.nan}}).ffill()
    #df_chords = df_beats.replace({'new_chord': {'': np.nan}}).ffill()
    df_chords = df_beats.replace({'': np.nan}).ffill()

    # Define new index with the key (melid, bar, beat)
    new_index = ['melid', 'bar', 'beat']
    df_chords_new = df_chords.set_index(new_index, drop=True)
    df_melody_new = df_melody.set_index(new_index, drop=True)

    # Concatenate the dataframes using the new index and then reset the index again
    #df_beats_mel = pd.concat([df_melody_new, df_chords_new.reindex(df_melody_new.index)], axis=1)
    df_beats_mel = df_chords_new.merge(df_melody_new, left_on=new_index, right_on=new_index, how='outer')
    df_beats_mel = df_beats_mel.reset_index(drop=False)

    return df_beats_mel