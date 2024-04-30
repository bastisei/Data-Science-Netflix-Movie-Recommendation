# Netflix Movie Recommendation - Collaborative Based

This project aims to build a recommendation system for Netflix movies using collaborative filtering techniques. Collaborative filtering is a method used by recommendation systems to make automatic predictions about the interests of a user by collecting preferences from many users.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [Source](#source)
- [Organization](#organization)

## Introduction

The goal of this project is to provide personalized movie recommendations to users based on their past ratings and preferences. The recommendation system is built using collaborative filtering algorithms, specifically Singular Value Decomposition (SVD) and Non-Negative Matrix Factorization (NMF).

## Features

- Train and evaluate collaborative filtering models (SVD and NMF).
- Generate top movie recommendations for users.
- Cross-validation to evaluate model performance.
- Save and load trained models for future use.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bastisei/Data-Science-Netflix-Movie-Recommendation.git

2. Navigate to the project directory: 
   ```bash
   cd Data-Science-Netflix-Movie-Recommendation

3. Install the required dependencies: 
   ```bash
   pip install -r requirements.txt

## Usage

1. Train and evaluate the SVD model: 
   ```bash
   python src/models/train_model.py --model SVD

2. Train and evaluate the NMF model: 
   ```bash
   python src/models/train_model.py --model NMF

3. Generate top movie recommendations for a user: 
   ```bash
   python src/models/predict_model.py --user_id <user_id>

## Contributing

Contributions are welcome! If you would like to contribute to this project, feel free to submit a pull request.

## Source

Data Source: https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset/data

## Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
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
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------
