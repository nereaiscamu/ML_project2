# Improving chord prediction in Jazz music using note information

This project is part of CS-433 Machine Learning @ EPFL.

Group Members:
- Nerea Carbonell
- Maximilian Gangloff
- Daniel Morales

## Project Structure

    .
    ├── data                                # Contains the file to preprocess the data
    │   ├── datasets                        # Contains the differenct encoded datasets
    │   |   ├── dataset1.pickle             # Only chords
    │   |   ├── dataset4.pickle             # Chords + Melody
    │   |   ├── dataset5.pickle             # Chords + Bass
    │   |   ├── dataset6.pickle             # Chords + Melody + Bass
    │   |   ├── dataset7.pickle             # Chords + Melody + Duration weighted
    │   |   ├── dataset8.pickle             # Chords + Melody weighted
    │   ├── Chord_vocab2.py                 # Creates .csv files that represent the chord vocabulary
    │   ├── combine_melody_beats.py         # Combines melody with beats table
    │   ├── dataset.py                      # Defines the torch Datasets
    │   ├── multi_hot_encoding.py           # Creates the different Datasets
    │   ├── one_hot_encoding.py             # 
    │   ├── wjazzd.db                       # Our final submission

    ├── models                              # 
    │   ├── trained_models                  # 
    │   |   ├── optimized_192_2_dataset_1   # 
    │   |   ├── optimized_192_2_dataset_4   # 
    │   |   ├── optimized_192_2_dataset_5   #
    │   |   ├── optimized_192_2_dataset_6   # 
    │   |   ├── optimized_192_2_dataset_7   # 
    │   |   ├── optimized_192_2_dataset_8   # 
    │   ├── lstm_chord_models.py            # 
    │   ├── lstm_melody_models.py           # 
    │   ├── chord_map.pkl                   # 

    ├── report                              # Location of the report folder
    │   ├── JazzChordPredictionLSTM_MND.pdf # Generated .pdf file of our report
    
    ├── scripts                             # 
    │   ├── costs.py                        # Cost functions
    │   ├── cross_val_helpers.py            # Helpers for cross_validation.py
    │   ├── implementations.py              # Implementations of ML methods requested
    │   ├── plots.py                        # Visualizations
    │   ├── proj1_helpers.py                # Helpers
    
    ├── README.md                           # README
    
    └── cross_validation.py                 # Perform hyperparameter search with k-fold cross-validation
    
    └── helpers.py                          # Data cleaning, transformation and augmentation

    └── hp_search.py                        # Data cleaning, transformation and augmentation

    └── load_model.py                       # Train our best model and generate submission
    
## Running

Firstly, make sure to download the train and test data of the challenge at https://www.aicrowd.com/challenges/epfl-machine-learning-higgs and put it in the ```./data``` foler. The files should be named ```test.csv``` and ````train.csv```.

Our best model is trained by ridge regression, achieving an 82.6% test accuracy. For full details on model selection as well as data processing, please check the report.

To train the model and generate predictions on test data, execute:

``` 
python run.py
```
The prediction will then be saved to the file ```submission.csv``` that can be found under ```./data/submission.csv```.