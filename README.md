# ML_project_EPFL
> - Project for CS-433 Machine Learning at EPFL
> - Prediction model about heart disease created by Team "mediterraneanOS"

## Table of Contents
1. [Introduction](#introduction)
2. [Setup](#setup)
3. [Project Structure](#project-structure)
4. [Usage](#usage)
5. [Authors](#authors)

## Introduction
This project consists of the implementation of a variety of regression techniques used to train models that will predict if a given patient has a heart condition or not. Apart from the basic functions, a customized version of logistic regression has been used to achieve better results. The datasets used for training and testing come from https://www.cdc.gov/brfss/annual_data/annual_2015.html, which is the Centers for Disease Control and Prevention, the national public health agency of the United States.

## Setup
- Clone the repository with `git clone https://github.com/CS-433/ml-project-1-mediterraneanos.git`.
- Follow the instructions below

## Project Structure
The `.csv` files should be stored `/datasets` directory, the rest of the source code is in the root directory, since there is only five source code files.

## Usage
**Datasets**:
- Put in the `/datasets` directory the `.csv` files that are going to be used for training.
- The datasets should be named `x_test.csv`, `x_train.csv` and `y_train.csv`.

**Hyperparameter tuning**:
Multiple parameters in `run.ipynb` can be adjusted to fine tune the performance of the model. These parameters are:
- `nans_threshold`: regulates the proportion of NaN values admitted per column to be added for training.
- `decay_rate`: the rate at which the learning rate decays over iterations.
- `degree_vector`: a list of degrees to be used for polynomial feature expansion.
- `max_iters`: the number of iterations that the gradient descent algorithm will run.
- `gamma_values`: a list of learning rates to be tested.
- `lambda_values`: a list of regularization parameters to be tested.
- `batch_size`: the size of the mini-batches used in stochastic gradient descent.

**Running the program**:
Run the code in `run.ipynb` and wait for the results and the visualizations.

## Authors

| Student's name | SCIPER |
| -------------- | ------ |
| Leonardo De Novellis | 386074 |
| Luis Bustamante | 366705 |
| Antonio Silvestri | 376875 |
