# !/usr/bin/env python
# coding: utf-8
# Developer:  Elliott Wobler
# University of Luxembourg, Interdisciplinary Space Master
# Machine Learning, October 2022
import json
import pickle
import argparse
import sklearn.model_selection as ms
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Local file with data and model evaluation functions:
from shared_functions import *

# # ConvergenceWarning prints to console are super noisy
# # so import these libraries to disable them
# # https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning


def neural_network_classifier(X_train, y_train, X_test):
    """
    Implement the MLPClassifier,
    fit the data, and
    capture the predicted results
    """
    NN = MLPClassifier(
        activation="logistic",
        # Logistic Activation Function:
        # "This function takes any real value as input and outputs values in the range of 0 to 1.
        # The larger the input (more positive), the closer the output value will be to 1.0,
        # whereas the smaller the input (more negative), the closer the output will be to 0.0, as shown below.""
        # - https://www.v7labs.com/blog/neural-networks-activation-functions
        learning_rate="invscaling",
        # ‘invscaling’ gradually decreases the learning rate at each time step ‘t’
        solver="adam",
        # Adam (Adaptive Moment Estimation)
        # "An algorithm for optimization technique for gradient descent.
        # The method is really efficient when working with large problem involving a lot of data or parameters.
        # It requires less memory and is efficient.
        # Intuitively, it is a combination of the ‘gradient descent with momentum’ algorithm and the ‘RMSP’ algorithm."
        # - https://www.geeksforgeeks.org/intuition-of-adam-optimizer/
        hidden_layer_sizes=(200,),
        alpha=0.0001,
        max_iter=800,
    )
    # print(NN.get_params())
    NN.fit(X_train, y_train)
    # Use the model
    y_pred = NN.predict(X_test)
    return y_pred, NN


def random_forest_classifier(X_train, y_train, X_test):
    """
    Implement the RandomForestClassifier,
    fit the data, and
    capture the predicted results
    """
    RF = RandomForestClassifier(
        n_estimators=400,
        min_samples_split=5,
        min_samples_leaf=1,
        max_features="sqrt",
        max_depth=50,
        bootstrap=False,
    )
    # print(RF.get_params())
    RF.fit(X_train, y_train)
    # Use the model
    y_pred = RF.predict(X_test)
    return y_pred, RF


def get_n_options(step, max, tuples=False):
    """
    Helper function to generate an array of values/tuples
    for cross validation hyperparameter options
    (e.g. MLP `hidden_layer_sizes` or RandomForest `n_estimator`)
    """
    vals = []
    for x in range(step, max, step):
        if tuples == True:
            vals.append((x,))
        else:
            vals.append(x)
    return vals


def fit_and_score_cv(validator, X_train, y_train, X, y, cv):
    """
    Capture the best performing parameters and accuracy score
    during the cross validation process
    """
    validator.fit(X_train, y_train)
    scores = ms.cross_val_score(validator.best_estimator_, X, y, cv=cv)
    print("\nCV Scores:\n", scores)
    best_params = "Best parameters found: {}\n".format(validator.best_params_)
    accuracy = "%0.2f accuracy with a standard deviation of %0.2f\n" % (
        scores.mean(),
        scores.std(),
    )
    return best_params, accuracy


def cross_validate(name, model, X_train, y_train, X, y, param_grid):
    """
    Use cross validation methods to determine the optimal hyperparameters
    and output the results to a file for manual analysis
    """
    print("{}Running {} Cross Validation...{}".format(BOLD, name, UNBOLD))
    cv = None  # `None` = use the default 5-fold cross validation
    njobs = 4  # number of jobs to run in parallel (-1 for "all")
    # Perform Grid Search cross validation
    grid = ms.GridSearchCV(model, param_grid, n_jobs=njobs, cv=cv)
    gp, ga = fit_and_score_cv(grid, X_train, y_train, X, y, cv)
    # Perform Randomized Search cross validation
    rando = ms.RandomizedSearchCV(model, param_grid, n_jobs=njobs, cv=cv)
    rp, ra = fit_and_score_cv(rando, X_train, y_train, X, y, cv)
    # Save cross validation results to disk
    with open(
        "cross_validation_results/{}_Cross_Validation_Scores.txt".format(name), "w"
    ) as f:
        f.writelines(["[ Grid Search ]\n\n", gp, ga])
        f.write("\n- - - - - - - - - - - - - - - -\n")
        f.writelines(["\n[ Randomized Search ]\n\n", rp, ra])
        f.write("\n- - - - - - - - - - - - - - - -\n")
        f.write("\nparam_grid:\n\n")
        f.write(json.dumps(param_grid))


def save_model_to_disk(name, model):
    """
    Save a trained model to disk
    so it can be loaded and used again in the future
    """
    filename = "saved_models/final_{}_model.sav".format(name)
    pickle.dump(model, open(filename, "wb"))
    print("Saved model: `{}`".format(filename))
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cv-mlp",
        action="store_true",
        help="Peform cross validation (hyperparameter tuning) for MLP model",
    )
    parser.add_argument(
        "--cv-rf",
        action="store_true",
        help="Peform cross validation (hyperparameter tuning) for RandomForest model",
    )
    args = parser.parse_args()
    # Load the training dataset (70% of total)
    # [ Rows 1 to 7000 (70%) of `dataset.csv` ]
    data = load_data("data/train.csv")
    # Clean, Transform, and Extract Labels
    X, y = label_extraction(data)
    # Split the Data into Train and Test Sets
    # with 90% for training set and 10% for testing set
    X_train, X_test, y_train, y_test = split_data(X, y, 0.1)
    # Scale the Features
    X_train, X_test = feature_scaling(X_train, X_test)
    # Model Training
    nn_y_pred, NN = neural_network_classifier(X_train, y_train, X_test)
    rf_y_pred, RF = random_forest_classifier(X_train, y_train, X_test)
    # Model Evaluation
    evaluate_model("Neural_Network", "saved_model_evaluations", y_test, nn_y_pred)
    evaluate_model("Random_Forest", "saved_model_evaluations", y_test, rf_y_pred)
    # Save Models
    saved_nn = save_model_to_disk("neural_network", NN)
    saved_rf = save_model_to_disk("random_forest", RF)
    ######################
    # MLP Cross Validation
    ######################
    if args.cv_mlp == True:
        cross_validate(
            "Neural_Network",
            NN,
            X_train,
            y_train,
            X,
            y,
            [
                {
                    "activation": ["logistic", "tanh", "relu"],
                    "alpha": [0.0001, 0.3, 0.9],
                    "hidden_layer_sizes": get_n_options(100, 500, True),
                    "learning_rate": ["constant", "adaptive", "invscaling"],
                    "max_iter": [200, 500, 800],
                    "solver": ["lbfgs", "sgd", "adam"],
                }
            ],
        )
    # ###############################
    # # RandomForest Cross Validation
    # ###############################
    if args.cv_rf == True:
        cross_validate(
            "Random_Forest",
            RF,
            X_train,
            y_train,
            X,
            y,
            [
                {
                    "bootstrap": [True, False],
                    "max_depth": [10, 50, 100, None],
                    "max_features": ["auto", "sqrt"],
                    "min_samples_leaf": [1, 2, 4],
                    "min_samples_split": [2, 5, 10],
                    "n_estimators": get_n_options(100, 500),
                }
            ],
        )
