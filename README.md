# Breast Cancer Prediction


## Exploratory Data Analysis and Data Processing

In this step I used the explore_data function in main.py to quickly go through and manually look at possible correlations between the different variables and diagnosis.

I could have probably used some statistics or something to do this part but due to time I just did manually. This part was very subjective so I could have definitley made some mistakes here, but overall I think I was able to catch a lot of very strong correlations.

TODO: Important variables with charts
### Important Variables I found:
#### radius_mean
    - Larger radius seems to mean Malignant
#### 

## The logistic regression model

I have no prior experience with any models like this, but after a quick search the logistic regression model seems to be the easiest for this application. So this is what I started with.

### First Results with no tuning:

Accuracy Score: 0.956140350877193

TODO: Confusion matrix

TODO: Analysis

### Tuning Hyperparameters

I wanted to try and scale up the process of tuning the hyperparameters of the logistic regression model, so I looked in the documentation for LogisticRegression in scikit-learn to see what I could mess with.

#### solver

By default LogisticRegression uses lbfgs

