# Breast Cancer Prediction


## Exploratory Data Analysis and Data Processing

In this step I used the explore_data function in main.py to quickly go through and manually look at possible correlations between the different variables and diagnosis.

I could have probably used some statistics or something to do this part but due to time I just did manually. This part was very subjective so I could have definitley made some mistakes here, but overall I think I was able to catch a lot of very strong correlations.

TODO: Important variables with charts
### Correlated variables that I found:
#### radius_mean

## The model tuner

To be honest, I have no prior experience with any type of model training like this. So while I could research each hyperparameter in each model and try and see what would work best for this situation, I thought rolling out my own "model tuner" would be both more fun and effective.

The code is contained in `tuner.py` and it can be universally used to mass-test hyperparameters to optimize models via the `tune` function. 

It is highly configurable and can essentially test all the possible combinations of set parameters to model using multiprocessing to scale up and make things faster. 

This "model tuner" also can help me see if excluding certain variables can increase my model's accuracy. This could also be used to solve my problem of subjectivity in the Exploratory Data Analysis portion.


## The logistic regression model

I have no prior experience with any models like this, but after a quick search the logistic regression model seems to be the easiest for this application. So this is what I started with.

### First Results with no tuning:

#### Accuracy Score: 0.956140350877193

![Log Regress 1 Image](/confusion/log_regress_1.jpg)

TODO: Analysis
