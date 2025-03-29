#!/bin/python3

import pandas
import basic
import matplotlib.pyplot as plt

import argparse

# I used this to find correlation between the variables
# Just shows a bar chart comparing the diagnosis to the variable value
# If I found a correlation it would automatically save the chart into the ./correlations/ directory
def explore_data(data, n=30):
    for column in data.columns[2:]:
        chart = data.head(n).plot(kind='bar',x='diagnosis',y=column,figsize=(10,10),title=column)
        plt.show()
        isCorrelated = input('If you found a possible correlation, type "y", type "e" to exit: ')
        if isCorrelated == "y":
            plt.savefig('./correlations/%s' % (column))
        if isCorrelated == "e":
            break
    print("All done!")

# Load the csv file into pandas
def load_dataset():
    config = basic.readconfig('main')
    data = pandas.read_csv(config['dataset_file'])
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Breast Cancer Prediction',
                    description='CHS Club Hackathon')
    parser.add_argument('command')
    args = parser.parse_args()
    if args.command == "explore":
        explore_data(load_dataset())
