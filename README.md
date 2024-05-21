[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/LkCf-P6F)
Revolutionizing Bird Sound Classification: Leveraging Machine Learning and Deep Learning Techniques
==============================
# DSEI210-S24-Final-Project
Final Project for DSEI210-S23 

This repo contains the code and resources for our project focused on recognizing bird species based on their sounds. We utilized various features and models, culminating in the development of a CNN + RNN model with Mel spectrogram features, which achieved the best performance.

## Dataset
The dataset can be downloaded from kaggle. Link to download: https://www.kaggle.com/datasets/vinayshanbhag/bird-song-data-set?resource=download
The link can also be found at this path in the repo: data/raw/Dataset. 

## Code
The code to reproduce the entire project can be found under the src folder. Requirement.txt file lists all the Python dependencies required to run the code for the entire project. 

**Instructions on running the code for the Best Model (CNN+RNN)**
-> Please download and extract the dataset to a folder. The extracted folder will have a "bird_songs_metadata.csv" file and a "wavfiles" folder containing all the wav files of bird sounds.
-> Please download the .py files below to a folder.
src/data/make_dataset_deep.py
src/features/build_features_deepmodels.py
src/models/train_model_deep.py
src/models/predict_model_deep.py
src/visualization/visualisations_deep_models.py
src/run_bestmodel.py
-> Please edit the dataset paths in lines #19 and #20 of run_bestmodel.py to the folder paths containing the downloaded data. 
-> Run the run_bestmodel.py file. Please note it will take a very long time to train if there isn't a GPU available. However, the trained model can be downloaded from here: models/3CNN_1RNN_regularized_model.zip. Extract the model to a folder and edit line # 33 of the run_bestmodel.py file with the extracted model path. Ensure to comment out lines 26-31 which are related to model creation and training.
-> The best model code can also be run on a notebook. Please refer to this notebook to reproduce the results: notebooks/Run_BestModel_ExampleNB_AdiM.ipynb












Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third-party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. The naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results-oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
