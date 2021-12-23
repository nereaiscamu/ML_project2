# Improving chord prediction in Jazz music using note information

This project is part of CS-433 Machine Learning @ EPFL.

Group Members:
- Nerea Carbonell
- Maximilian Gangloff
- Daniel Morales

## Project Structure


    ├── data                                # Contains the file to preprocess the data
    │   ├── combine_melody_beats.py         # Combines melody with beats table
    │   ├── dataset.py                      # Defines the torch Datasets
    │   ├── multi_hot_encoding.py           # Creates the different Datasets
    │   ├── wjazzd.db                       # Original Weimar Jazz Database
    │   ├── datasets						# Folder to save the datasets when training/loading the models    

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

    
    ├── JazzChordPredictionLSTM_MND.pdf     # Generated .pdf file of our report
    └── README.md                           # README
    └── cross_validation.py                 # Uses k-fold cross-validation to get average accuracy
    └── helpers.py                          # Contains helper functions for the train and cross_validation file such as calculating the loss and accuracy
    └── hp_search.py                        # Uses random or grid search to find best hyperparameters
    └── load_model.py                       # Loads a selected model
    └── qualitative_analysis.py             # Analysis between Baseline and Melody
    └── Tune Accuracy comparison.csv        # Accuracy comparison of the Baseline and Melody for each test sample (song)
    └── train.py                            # Trains a given model
	
## Run the models

To check the different models' accuracy, load the models and generate the datasets by running ```run.py```. There, every dataset and model will be loaded from its generated and saved ```.pickle``` and ```.pth``` files. Then, the accuracies will be calculated and displayed in the terminal resulting in the following table.

| Model           | Test accuracy |
|-----------------|:-------------:|
| Baseline        | 50.89 %       |
| Melody          | 53.93 %       |
| Melody weighted | 53.47 %       |
| Melody duration | 53.26 %       |
| Bass            | 52.72 %       |
| Melody + Bass   | 53.68 %       |


## Training models

To train one of the models, from the command line type ```train.py --dataset=dataset_id```, where dataset_id is the index preceding the different models listed in trained models. Also, if the dataset has been previously generated, add as an argument ```--use-saved-dataset=True```.

The following arguments can be passed to ```train.py```:

| Argument        | Description   |
|-----------------|-------------|
| --dataset       | Integer between 1 and 6 to choose the model   |
| --hidden-dim   | Hidden layers, default is 192 which is the optimal for all models       |
| --lstm-layers | LSTM layers, default is 2 which is the best for all models      |
| --max-epochs | Maximum number of epochs      |
| --early-stopping            | Number of iteration before stopping after no improvement was found      |
| --seed   | Seed       |
| --lr   | Learning rate       |
| --wd   | Weight decay       |
| --dropout   | Dropout       |
| --use-saved-dataset   | Use saved dataset       |
| --save-path   | Path of the model to be saved       |


## Qualitative analysis

To generate the plots and tables used in the qualitative analysis, run qualitative_analysis.py. The default models to compare are 1_Baseline and 2_Melody. 
