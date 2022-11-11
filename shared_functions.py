# !/usr/bin/env python
# coding: utf-8
# Developer:  Elliott Wobler
# University of Luxembourg, Interdisciplinary Space Master
# Machine Learning, Fall 2022
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# For print output readability
BOLD = "\033[1m"
UNBOLD = "\033[0m"


def load_data(filename="data/dataset.csv"):
    """
    Load a CSV data file into a Pandas DataFrame
    and clean it up by dropping any missing values
    """
    # train = pd.read_csv("data/train.csv")  # 1 to 7000 (70%) of `dataset.csv`
    # test = pd.read_csv("data/test.csv")  # 7001 to 10000 (30%) of `dataset.csv`
    data = pd.read_csv(filename)
    # Clean data by removing missing values
    data = data.dropna()
    return data


def label_extraction(data):
    """
    Separate data features and labels
    after creating a new "label" vector of mapped integer values
    """
    # Create an ordered array of all possible "class" values,
    # determined by running `print(data["class"].unique())`
    categories = ["QSO", "GALAXY", "STAR"]
    # Ensure category order is maintained
    ordered = True
    # For classification, transform the "class" column of string values
    # into a "label" column of integer values: QSO=0, GALAXY=1, STAR=2
    data["label"] = pd.Categorical(
        data["class"], categories=categories, ordered=ordered
    ).codes
    # Separate the data features from the data labels
    X = data.drop(columns=["class", "label"])  # features
    y = data["label"]  # labels
    return X, y


def split_data(X, y, test_size):
    """
    Split the data into separate training and test sets
    """
    (X_train, X_test, y_train, y_test) = ms.train_test_split(
        X, y, test_size=test_size, random_state=0
    )
    return X_train, X_test, y_train, y_test


def feature_scaling(X_train, X_test):
    """
    Scale/transform the feature data
    """
    sc = StandardScaler()
    # Scale the training data
    # while also learning the scaling parameters
    # (i.e. mean and variance of the features)
    X_train = sc.fit_transform(X_train)
    # Just scale the test data
    X_test = sc.transform(X_test)
    return X_train, X_test


def evaluate_model(name, dir, y_test, y_pred):
    """
    Evaluate model performance using:
    - the accuracy score
    - the confusion matrix
    - the classification report (precision, recall, f1-score, support)
    """
    outfile = "{}/{}_Model_Evaluation.txt".format(dir, name)
    confmat = confusion_matrix(y_test, y_pred)
    with open(outfile, "w") as f:
        f.write("[ {} Evaluation ]\n\n".format(name))
        f.write("Accuracy Score: {}\n\n".format(accuracy_score(y_test, y_pred)))
        f.write("Confusion Matrix:\n{}\n\n".format(confmat))
        f.write(
            "Classification Report:\n{}\n".format(classification_report(y_test, y_pred))
        )
        # Note: F1 scores range from 0 to 1, with higher scores being generally better
        print("Model evaluation report saved to `{}`".format(outfile))
    ConfusionMatrixDisplay(confmat).plot()
    plt.savefig("{}/{}_Confusion_Matrix_Plot.png".format(dir, name))
