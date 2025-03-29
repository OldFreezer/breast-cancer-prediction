#!/bin/python3

import pandas
import basic

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

# Tests the logic_regress model and returns the confusion matrix for visualization
def test_model(dataset, exclude_variables=[], random_state=16, test_size=0.25, max_iter=5000, solver='liblinear'):
    config = basic.readconfig('main')
    vars = config['important_variables']
    for exclude in exclude_variables:
        vars.remove(exclude)
    X = dataset[vars] # Independant Variables
    
    Y = dataset['diagnosis'] # Dependant Variables

    # Randomly split the dataset into a training and testing set
    # From what I understand, random_state is essentially a seed for reproducability
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state) # Uses a 3:1 ratio training:testing

    # I had to increase the max_iter to 3000 in order to get the model to converge
    logreg = LogisticRegression(random_state=random_state,max_iter=max_iter,solver=solver)

    # Fit the model with the data
    logreg.fit(X_train, y_train)

    # Predict the dependant variables using the test data
    y_pred = logreg.predict(X_test)

    # Create a confusion matrix, which basically evaluates the performance of the model
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Get the accuracy score
    score = metrics.accuracy_score(y_test,y_pred)

    return cnf_matrix, score

SOLVERS = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]