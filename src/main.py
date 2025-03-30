#!/bin/python3

import pandas
import basic
import matplotlib.pyplot as plt
import seaborn as sns
import numpy
import json
import os
import tuner

import argparse

import models

# I used this to find correlation between the variables
# Just shows a bar chart comparing the diagnosis to the variable value
# If I found a correlation it would automatically save the chart into the ./correlations/ directory
def explore_data(data, n=30):
    important_vars = []
    for column in data.columns[2:]:
        chart = data.head(n).plot(kind='bar',x='diagnosis',y=column,figsize=(10,10),title=column)
        plt.savefig('./correlations/%s' % (column))
        plt.show()
        isCorrelated = input('If you found a possible correlation, type "y", type "e" to exit: ')
        if isCorrelated == "y":
            important_vars.append(column)
        if isCorrelated == "e":
            break
    config = basic.readconfig('main')
    config['important_variables'] = important_vars
    basic.writef('./config/main.config',json.dumps(config))
    print("Wrote all possible import!")

# Cleans all uncorrelated variable charts from ./correlations/
def clean_uncorrelated(important_vars):
    for file in os.listdir('./correlations/'):
        name = file.replace('.png','')
        if name not in important_vars:
            os.remove('./correlations/%s' % (file))

# Visualize a confusion matrix via a heatmap
def visualize_confusion(cnf_matrix):
    class_names=[0,1] # name  of classes
    fig, ax = plt.subplots()
    tick_marks = numpy.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    # create heatmap
    sns.heatmap(pandas.DataFrame(cnf_matrix), annot=True, cmap="Greens" ,fmt='g')
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title('Confusion matrix', y=1.1)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Load the csv file into pandas
def load_dataset(clean=False):
    config = basic.readconfig('main')
    data = pandas.read_csv(config['dataset_file'])
    # Clean out all uncorrelated variables
    if clean:
        alsoImportant = ["id", "diagnosis"] # Variables we don't want to remove but will not be used in the model
        for column in data.columns:
            if column not in config['important_variables'] and column not in alsoImportant:
                data.drop(column,axis=1,inplace=True)
    return data

if __name__ == "__main__":
    #explore_data(load_dataset())
    parser = argparse.ArgumentParser(
                    prog='Breast Cancer Prediction',
                    description='CHS Club Hackathon')
    parser.add_argument('command')
    args = parser.parse_args()
    if args.command == "explore":
        explore_data(load_dataset())
    elif args.command == "cleanCharts": 
        clean_uncorrelated(basic.readconfig('main')['important_variables'])
    elif args.command == "tuneLR":
        tuner.tune(
            models.logic_regress,
            load_dataset(clean=True),
            basic.readconfig('main')['important_variables'],
            params={
                "test_size": 0.25, # Messing with the test size can skew results
                "solver": "liblinear",
                "exclude_variables": {"type": list, "len": 3, "vals": basic.readconfig('main')['important_variables']},
                "C": {"type": float, "min": 0.1, "max": 1.1, "interval": 0.1}
            }
        )
    elif args.command == "testLR":
        # Test best logic regress model
        cnf_matrix, score = models.logic_regress(
            load_dataset(clean=True),
            exclude_variables=["texture_mean","area_se","texture_worst"],
            C=0.6,
            test_size=0.25,
            random_state=None,
            solver="liblinear"
            )
        print('Accuracy Score: %s' % (score))
        visualize_confusion(cnf_matrix)
    elif args.command == "tuneDT":
        tuner.tune(
            models.decision_tree,
            load_dataset(clean=True),
            basic.readconfig('main')['important_variables'],
            params={
                "test_size": 0.25,
                "exclude_variables": ["texture_mean", "area_se", "texture_worst"],
                "criterion": {"type": str, "vals": ["gini", "entropy", "log_loss"]},
                "max_depth": {"type": int, "min": 1, "max": 5, "interval": 1},
                "min_samples_split": {"type": int, "min": 2, "max": 10, "interval": 1}
            }
        )
    elif args.command == "testDT":
        # Test Decision Tree model
        cnf_matrix, score = models.decision_tree(
            load_dataset(clean=True),
            showPlot=True,
            criterion="gini",
            max_depth=4,
            min_samples_split=4,
            test_size=0.25,
            exclude_variables=[
                "texture_mean",
                "area_se",
                "texture_worst"
            ]
            )
        print('Accuracy Score: %s' % (score))
        visualize_confusion(cnf_matrix)
    else:
        # TODO: Print help
        print('Command not found')