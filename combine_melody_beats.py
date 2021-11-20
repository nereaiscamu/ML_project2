import numpy as np
import pandas as pd


def combine_melody_beats(df_melody, df_beats):
    """
    Fills the chords in the beats table and combines the melody table and the beats table 
    using the MultiIndex key of (melid, bar, beat).

    Args:
        df_melody: Dataframe of the melody table
        df_beats: Dataframe of the beats table

    Returns:
        Dataframe consisting of the combination of both tables
    """
    df_chords = df_beats.replace({'chord': {'': np.nan}}).ffill()

    # Define new index with the key (melid, bar, beat)
    df_chords_new = df_chords.set_index(['melid', 'bar', 'beat'], drop=False)
    df_melody_new = df_melody.set_index(['melid', 'bar', 'beat'], drop=False)

    # Concatenate the dataframes using the new index and then reset the index again
    df_mel_beats = pd.concat([df_melody_new, df_chords_new.reindex(df_melody_new.index)], axis=1)
    df_mel_beats = df_mel_beats.reset_index(drop=True)

    # Remove duplicate columns
    # Duplicated columns are:
    # - melid
    # - onset
    # - bar
    # - beat
    df_mel_beats = df_mel_beats.loc[:,~df_mel_beats.columns.duplicated()]

    return df_mel_beats