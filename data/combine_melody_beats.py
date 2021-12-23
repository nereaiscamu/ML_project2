import numpy as np

def merge_melody_beats(df_melody, df_beats):
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

def encode_melody(df_melody, df_beats, pitch_sequence=False):
    """
    Encodes the pitches of the melody in combination with the beats.

    Args:
        df_melody: Dataframe of the melody table
        df_beats: Dataframe of the beats table
        pitch_sequence: Boolean to use pitch per chord (False) or sequence of chord (True) 

    Returns:
        Dataframe containing the encoded pitches
    """

    df_beats_mel = merge_melody_beats(df_melody, df_beats)
    
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
    df_beats_mel['chord_changed'] = (df_beats_mel['new_chord'].shift() != df_beats_mel["new_chord"]).cumsum()

    # Group chord changes to get sequences
    pitch_sequences = [g['pitch_encoded'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]
    bass_pitch_sequences = [g['bass_pitch_encoded'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]
    duration_sequence = [g['duration'].tolist() for k, g in df_beats_mel.groupby('chord_changed')]

    # Identify last row of current chord
    df_beats_mel['pitch_sequence'] = (df_beats_mel['new_chord'].shift(-1) != df_beats_mel["new_chord"])

    # Change type to type object to add list to cell
    df_beats_mel['pitch_sequence'] = df_beats_mel['pitch_sequence'].astype(object)
    df_beats_mel['bass_pitch_sequence'] = df_beats_mel['pitch_sequence']
    df_beats_mel['duration_sequence'] = df_beats_mel['pitch_sequence']

    # Set sequence to last played chord
    for idx, _ in df_beats_mel.iterrows():
        if df_beats_mel.at[idx, 'pitch_sequence'] == True:
            df_beats_mel.at[idx, 'pitch_sequence'] = pitch_sequences.pop(0)
            df_beats_mel.at[idx, 'bass_pitch_sequence'] = bass_pitch_sequences.pop(0)
            df_beats_mel.at[idx, 'duration_sequence'] = duration_sequence.pop(0)

    df_beats_mel.drop(['bass_pitch_encoded', 'chord_changed'], axis=1, inplace=True)
    df_beats_mel.drop(df_beats_mel[df_beats_mel['pitch_sequence'] == False].index, inplace=True)
    
    return df_beats_mel