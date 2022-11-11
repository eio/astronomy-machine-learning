# !/usr/bin/env python
# coding: utf-8
# Developer:  Elliott Wobler
# University of Luxembourg, Interdisciplinary Space Master
# Machine Learning, October 2022
import pandas as pd
import matplotlib.pyplot as plt

# Local file with data and model evaluation functions:
from shared_functions import *


def print_and_write(f, text):
    """
    Utility function for writing text to a provided output file
    and simultaneously printing output for the script user
    """
    print(text)
    # remove formatting before writing to file
    text = text.replace(BOLD, "")
    text = text.replace(UNBOLD, "")
    # write the text to the provided output file
    f.write(text + "\n\n")


def document_investigation(data):
    """
    Collect and print details about the loaded dataset
    """
    outfile = "data_inspection_output/data_details.txt"
    # Open an output file to document our investigation
    with open(outfile, "w") as f:
        # Print some data records, with the column headers
        count = 10
        head = "\n{}First {} data records:{} \n{}\n".format(
            BOLD, count, UNBOLD, data.head(count)
        )
        tail = "\n{}Last {} data records:{} \n{}\n".format(
            BOLD, count, UNBOLD, data.tail(count)
        )
        print_and_write(f, head)
        print_and_write(f, tail)
        # Print stats on empty values
        total_missing = data.isnull().sum().sum()
        if total_missing > 0:
            # If there are any empty values, print how many empty values per column
            missing = "{}Empty cells per column:{} \n{}\n".format(
                BOLD, UNBOLD, data.isnull().sum()
            )
            print_and_write(f, missing)

        # After finding the relevant label column, get some more information about it
        uniq_classes = "{}Unique labels/classes:{} \n{}\n".format(
            BOLD, UNBOLD, data["class"].unique()
        )
        print_and_write(f, uniq_classes)
        # Print more details about the "class" column
        class_details = "{}Label column details:{} \n{}\n".format(
            BOLD, UNBOLD, data["class"].describe()
        )
        print_and_write(f, class_details)
        # Print the count of each unique class
        class_counts = "{}Class counts:{} \n{}\n".format(
            BOLD, UNBOLD, data["class"].value_counts()
        )
        print_and_write(f, class_counts)


def visualize(data, columns, count=10000):
    """
    Visualize data for the specified column(s)
    """
    data = data.head(count)
    data[columns].plot()
    plt.gcf().savefig("data_inspection_output/{}.png".format(columns))
    plt.close()


def inspect_data(data):
    """
    Output text details and plot images
    to the `inspection_output/` directory
    """
    document_investigation(data)
    # Plot the "redshift" values for manual inspection
    visualize(data, "redshift")
    # Plot the ascension and declination values:
    # ra = J2000 Right Ascension (r-band)
    # dec = J2000 Declination (r-band)
    visualize(data, ["ra", "dec"])
    # Plot the Thuan-Gunn astronomic magnitude system for just 7,000 records.
    # u, g, r, i, z represent the response of the 5 bands of the telescope.
    visualize(data, ["u", "g", "r", "i", "z"], 7000)


if __name__ == "__main__":
    # Load the full dataset
    data = load_data("data/dataset.csv")
    # Inspect the dataset
    inspect_data(data)
