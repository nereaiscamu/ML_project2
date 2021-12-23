# Improving chord prediction in Jazz music using note information

This project is part of CS-433 Machine Learning @ EPFL.

Group Members:
- Nerea Carbonell
- Maximilian Gangloff
- Daniel Morales

## Project Structure

    .
    ├── Chord_Vocab                         # 
    │   ├── 3hot_mod.csv                    # 
    │   ├── 3hot.csv                        # 
    │   ├── 3hot.txt                        # 
    │   ├── Beats_Modifier                  # 
    │   ├── Chord_Dicitonary.csv            # Map of our chord dictionary as .csv file
    |   ├── chord_map.pkl                   # Map of the chord vocabulary
    |   ├── Chord_vocab.py                  # 

    ├── data                                # Contains the file to preprocess the data
    │   ├── Chord_vocab.py                  # Creates .csv files that represent the chord vocabulary
    │   ├── combine_melody_beats.py         # Combines melody with beats table
    │   ├── dataset.py                      # Defines the torch Datasets
    │   ├── multi_hot_encoding.py           # Creates the different Datasets
    │   ├── one_hot_encoding.py             # 
    │   ├── wjazzd.db                       # Our final submission

    ├── models                              # 
    │   ├── trained_models                  # 
    │   |   ├── 1_Baseline                  # Model containing only the encoded chords
    │   |   ├── 2_Melody                    # Encoded chords + Melody
    │   |   ├── 3_Bass                      # Encoded chords + Bass
    │   |   ├── 4_Melody_Bass               # Encoded chords + Melody + Bass
    │   |   ├── 5_Melody_Duration           # Encoded chords + Melody weighted with duration
    │   |   ├── 6_Melody_Weighted           # Encoded chords + Melody weighted w/o duration
    │   ├── lstm_chord_models.py            # Models for the Baseline
    │   ├── lstm_melody_models.py           # Models for the Melody encoding

    ├── report                              # Location of the report folder
    │   ├── JazzChordPredictionLSTM_MND.pdf # Generated .pdf file of our report

    ├── README.md                           # README
    └── cross_validation.py                 # Uses k-fold cross-validation to get average accuracy
    └── helpers.py                          # Contains helper functions for the train and cross_validation file such as calculating the loss and accuracy
    └── hp_search.py                        # Uses random or grid search to find best hyperparameters
    └── load_model.py                       # Loads a selected model
    └── qualitative_analysis.py             # Analysis between Baseline and Melody
    └── Tune Accuracy comparison.csv        # Accuracy comparison of the Baseline and Melody
    └── train.py                            # Trains a given model

## Running

