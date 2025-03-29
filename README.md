# Breast Cancer Prediction


## Exploratory Data Analysis and Data Processing

In this step I used the explore_data function in main.py to quickly go through and manually look at possible correlations between the different variables and diagnosis.

It sets up a bar graph for me to easily see if there are any outstanding differences between the malignant and benign diagnoses.

I could have probably used some statistics or something to do this part but due to time I just did manually. This part was very subjective so I could have definitley made some mistakes here, but overall I think I was able to catch a lot of very strong correlations.


### Some (Not All) Correlated Variables that I found:

![Log Regress 1 Image](/correlations/radius_worst.png)
![Log Regress 1 Image](/correlations/area_mean.png)
![Log Regress 1 Image](/correlations/perimeter_mean.png)

All of my correlated variables bar graphs are in ./correlations/

A list was placed in `main.config` that contains all of the important variables that I will use in model training:
```
["perimeter_se", "compactness_worst", "concavity_worst", "perimeter_worst", "compactness_mean", "texture_mean", "concave points_mean", "concave points_worst", "concavity_mean", "radius_se", "radius_worst", "area_worst", "area_se", "perimeter_mean", "radius_mean", "area_mean", "smoothness_worst", "texture_worst"]
```

## The model tuner

To be honest, I have no prior experience with any type of model training like this. So while I could research each hyperparameter in each model and try and see what would work best for this situation, I thought rolling out my own "model tuner" would be both more fun and effective.

The code is contained in `tuner.py` and it can be universally used to mass-test hyperparameters to optimize models via the `tune` function. 

It is highly configurable and can essentially test all the possible combinations of set hyperparameters using multiprocessing to scale up and make things faster. 

This "model tuner" also can help me see if excluding certain variables can increase my model's accuracy. This can be used to solve my problem of subjectivity in the Exploratory Data Analysis portion.

It is very simple and can be scaled up tremendously, but due to time constraints I'm not going to be able to play around with testing all the parameters on all the models. 

#### Side Note:
Any time that I talk about the `exclude_variables` model parameter, this is referring to which variables will be excluded from model training. 

## The logistic regression model

### First results with default options:

#### Accuracy Score: 0.956140350877193

![Log Regress 1 Image](/confusion/log_regress_1.png)

### Tuning:

After some quick searching, I found that the `liblinear` solver would be the best for a small amount of training data. So it was used along with a fixed 3:1 training:testing data ratio.

So I decided to only mess with which variables I should exclude and the C value.

Tuner Options:
```
{
    "test_size": 0.25, 
    "solver": "liblinear",
    "exclude_variables": {"type": list, "len": 3, "vals": IMPORTANT_VARIABLES},
    "C": {"type": float, "min": 0.1, "max": 1.1, "interval": 0.1}
}
```

#### Accuracy Score: 0.9824561403508771 
#### Parameters:
```
{
  "exclude_variables": [
    "texture_mean",
    "area_se",
    "texture_worst"
  ],
  "C": 0.6,
  "test_size": 0.25,
  "solver": "liblinear"
}
```

