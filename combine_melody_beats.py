import numpy as np
import pandas as pd


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

    df_mel_beats = concat_melody_beats(df_melody, df_beats)
    
    # Encode pitch
    df_mel_beats['pitch_encoded'] = np.mod(df_mel_beats['pitch'], 12)
    df_mel_beats['bass_pitch_encoded'] = np.mod(df_mel_beats['bass_pitch'], 12)

    df_mel_beats['pitch_encoded'] = df_mel_beats['pitch_encoded'].astype(int)
    df_mel_beats['bass_pitch_encoded'] = df_mel_beats['bass_pitch_encoded'].astype(int)

    ## Encode pitch for every chord of melody
    if not pitch_sequence:
        return df_mel_beats

    ## Encode sequence of pitch for every chord
    # Add column that represent chord changes
    df_mel_beats['chord_changed'] = (df_mel_beats['chord'].shift() != df_mel_beats["chord"]).cumsum()

    # Group chord changes to get sequences
    pitch_sequences = [g['pitch_encoded'].tolist() for k, g in df_mel_beats.groupby('chord_changed')]
    bass_pitch_sequences = [g['bass_pitch_encoded'].tolist() for k, g in df_mel_beats.groupby('chord_changed')]

    # Identify last row of current chord
    df_mel_beats['pitch_sequence'] = (df_mel_beats['chord'].shift(-1) != df_mel_beats["chord"])

    # Change type to type object to add list to cell
    df_mel_beats['pitch_sequence'] = df_mel_beats['pitch_sequence'].astype(object)
    df_mel_beats['bass_pitch_sequence'] = df_mel_beats['pitch_sequence']

    # Set sequence to last chord
    # TODO decrease running time if possible
    # TODO discuss where the sequence should be added
    for idx, _ in df_mel_beats.iterrows():
        if df_mel_beats.at[idx, 'pitch_sequence'] == True:
            df_mel_beats.at[idx, 'pitch_sequence'] = pitch_sequences.pop(0)
            df_mel_beats.at[idx, 'bass_pitch_sequence'] = bass_pitch_sequences.pop(0)

    # Drop useless columns and rows
    # TODO discuss if rows should be dropped or not
    df_mel_beats.drop(['pitch_encoded', 'bass_pitch_encoded', 'chord_changed'], axis=1, inplace=True)
    df_mel_beats.drop(df_mel_beats[df_mel_beats['pitch_sequence'] == False].index, inplace=True)

    return df_mel_beats


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
    df_melody = df_melody[['eventid', 'melid', 'pitch', 'bar', 'beat']]
    #df_beats = df_beats[['beatid', 'melid', 'bar', 'beat', 'chord', 'bass_pitch']]

    # Replace empty strings by nan and then use ffill to fill with last seen chord
    #df_chords = df_beats.replace({'chord': {'': np.nan}}).ffill()
    #df_chords = df_beats.replace({'new_chord': {'': np.nan}}).ffill()
    df_chords = df_beats.replace({'': np.nan}).ffill()

    # Define new index with the key (melid, bar, beat)
    df_chords_new = df_chords.set_index(['melid', 'bar', 'beat'], drop=False)
    df_melody_new = df_melody.set_index(['melid', 'bar', 'beat'], drop=False)

    # Concatenate the dataframes using the new index and then reset the index again
    df_mel_beats = pd.concat([df_melody_new, df_chords_new.reindex(df_melody_new.index)], axis=1)
    df_mel_beats = df_mel_beats.reset_index(drop=True)

    # Remove duplicate columns
    # Duplicated columns are:
    # - melid
    # - bar
    # - beat
    df_mel_beats = df_mel_beats.loc[:,~df_mel_beats.columns.duplicated()]

    return df_mel_beats