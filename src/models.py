#!/bin/python3

import pandas
import basic

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics, tree

# Test a model multiple times over and average out it's accuracy score
# I could multiprocess this, but it's not necessary for a low number of trials
def testModel(modelFunction, params, dataset, trials=30):
    scores = []
    params['random_state'] = None
    for i in range(0,trials):
        cnf_matrix, score = modelFunction(dataset, **params)
        scores.append(score)
    return sum(scores)/len(scores)

# Logistic Regression Model
def logic_regress(dataset, exclude_variables=[], random_state=16, test_size=0.25, max_iter=5000, C=1, solver='liblinear'):
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
    logreg = LogisticRegression(random_state=random_state,max_iter=max_iter,solver=solver,C=C)

    # Fit the model with the data
    logreg.fit(X_train, y_train)

    # Predict the dependant variables using the test data
    y_pred = logreg.predict(X_test)

    # Create a confusion matrix, which basically evaluates the performance of the model
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Get the accuracy score
    score = metrics.accuracy_score(y_test,y_pred)

    return cnf_matrix, score

# Decision Tree Model
def decision_tree(dataset, exclude_variables=[], random_state=16, test_size=0.25, showPlot=False, criterion="gini", max_depth=None, min_samples_split=2):
    config = basic.readconfig('main')
    vars = config['important_variables']
    for exclude in exclude_variables:
        vars.remove(exclude)
    X = dataset[vars] # Independant Variables
    
    Y = dataset['diagnosis'] # Dependant Variables

    # Randomly split the dataset into a training and testing set
    # From what I understand, random_state is essentially a seed for reproducability
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=random_state) # Uses a 3:1 ratio training:testing

    clf = tree.DecisionTreeClassifier(criterion=criterion,max_depth=max_depth,min_samples_split=min_samples_split)

    clf = clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    # Create a confusion matrix, which basically evaluates the performance of the model
    cnf_matrix = metrics.confusion_matrix(y_test, y_pred)

    # Get the accuracy score
    score = metrics.accuracy_score(y_test,y_pred)

    if showPlot:
        tree.plot_tree(clf)

    return cnf_matrix, score
