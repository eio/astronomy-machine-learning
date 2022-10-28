# !/usr/bin/env python
# coding: utf-8
# Developer:  Elliott Wobler
# University of Luxembourg, Interdisciplinary Space Master
# Machine Learning, October 2022
import pickle

# Local file with data and model evaluation functions:
from shared_functions import *


def load_model(name):
    """
    Load a trained model from disk
    so it can be used with new test data
    """
    loaded_model = pickle.load(
        open("saved_models/final_{}_model.sav".format(name), "rb")
    )
    return loaded_model


if __name__ == "__main__":
    # Load the holdout test dataset
    # which was not used at all during model training
    # to make sure the model is robust against overfitting
    # [ Rows 7001 to 10000 (30%) of `dataset.csv` ]
    data = load_data("data/test.csv")
    # Clean, Transform, and Extract Labels
    X, y = label_extraction(data)
    # Split the Data into Train and Test Sets
    # specifying 90% for test data, and 10% just for feature scaling
    X_train, X_test, y_train, y_test = split_data(X, y, 0.9)
    # Scale the Features
    X_train, X_test = feature_scaling(X_train, X_test)

    ################################
    # Test Saved MLP Neural Network
    ################################
    # Load the Saved Model
    model = load_model("neural_network")
    # Use the model
    nn_y_pred = model.predict(X_test)
    # Model Evaluation
    evaluate_model("Neural_Network", "loaded_model_evaluations", y_test, nn_y_pred)

    ###########################
    # Test Saved Random Forest
    ###########################
    # Load the Saved Model
    model = load_model("random_forest")
    # Use the model
    rf_y_pred = model.predict(X_test)
    # Model Evaluation
    evaluate_model("Random_Forest", "loaded_model_evaluations", y_test, rf_y_pred)
