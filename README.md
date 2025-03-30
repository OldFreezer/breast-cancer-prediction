# Breast Cancer Prediction

The goal of this project is to use machine learning models in order to predict malignant tumors in the dataset. I have never worked with SciKit or machine learning models in the past so this was a great learning experience for me.

Everything in this markdown file is put in chronological order. This is especially prevalent when I change my model tuner function due to an oversight, which while it forced me to re-run my tests, I still kept the findings from before I changed the tuner.

## Credits

I mostly used the scikit documentation for this project <https://scikit-learn.org/> and some help from StackOverflow and GeeksForGeeks. 

Generative AI was not used at all during this challenge.

## Exploratory Data Analysis and Data Processing

In this step I used the explore_data function in main.py to quickly go through and manually look at possible correlations between the different variables and diagnosis.

It sets up a bar graph for me to easily see if there are any outstanding differences between the malignant and benign diagnoses.

I could have probably used some statistics or something to do this part but due to time I just did it manually. This part was very subjective so I could have definitely made some mistakes here, but overall I think I was able to catch a lot of very strong correlations.

If I had enough time, I could have used my "Model Tuner" (See Below) to filter through which variables would work best with any given model. 

### Some (Not All) Correlated Variables that I found:

![Log Regress 1 Image](/images/correlations/radius_worst.png)
![Log Regress 1 Image](/images/correlations/area_mean.png)
![Log Regress 1 Image](/images/correlations/perimeter_mean.png)

All of my correlated variables bar graphs are in ./images/correlations/

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

#### Current Drawbacks

With all of my tests, I set the `random_state` seed to 16 by default, in order to make results deterministic. However it is possible that just by random chance a certain set of parameters scored a lot higher than the others. This could easily be fixed by setting the `random_state` to "None" and then running each set of possible parameters multiple times, then take the average of the score. Due to time constraints, I'm not able to do this so the "Model Tuner" stands as a proof-of-concept.

Due to a very small dataset there is the strong possibility of overfitting any of these models.

**EDIT:** After testing, I realized that this is a major flaw and I have improved the "Model Tuner" at the end of this README

#### Side Note:
Any time that I talk about the `exclude_variables` model parameter, this is referring to which variables (such as `radius_worst` or `texture_mean`) will be excluded from model training. 

## The logistic regression model

### First results with default options:

#### Accuracy Score: 0.956140350877193

![Log Regress 1 Image](/images/confusion/log_regress_1.png)

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

![Log Regress 2 Image](/images/confusion/log_regress_2.png)

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

## The Decision Tree Model

### First results with default options:

#### Accuracy Score: 0.9122807017543859

![DT 1 Image](/images/confusion/tree_1.png)

![DT 1 Image](/images/other/decision_tree_1.png)

I did not have time to actually analyze the decision tree plot, but I thought it would be useful to include.

### Tuning:

In order to maintain consistency with the previous model's training data, I decided to keep the same `exclude_variables`. So the only variables that I wanted to tune were `criterion`, `max_depth`, and `min_samples_split`

Tuner Options:
```
{
    "test_size": 0.25, 
    "exclude_variables": ["texture_mean", "area_se", "texture_worst"],
    "criterion": {"type": str, "vals": ["gini", "entropy", "log_loss"]},
    "max_depth": {"type": int, "min": 1, "max": 5, "interval": 1},
    "min_samples_split": {"type": int, "min": 2, "max": 10, "interval": 1}
}
```

#### Accuracy Score: 0.956140350877193

#### Parameters:
```
{
  "criterion": "gini",
  "max_depth": 4,
  "min_samples_split": 4,
  "test_size": 0.25,
  "exclude_variables": [
    "texture_mean",
    "area_se",
    "texture_worst"
  ]
}
```


![DT 2 Image](/images/confusion/tree_2.png)

![DT 2 Image](/images/other/decision_tree_2.png)


## Testing Both Models:

Now I had 2 models that were very crudely tuned and I wasn't comfortable with deciding on one without doing some more testing first. This was mostly due to the fact that the `random_state` variable was fixed, so I decided to run multiple tests of each "tuned" model side by side and average out their accuracy scores.

So I implemented the `testModel` function, which would run a model multiple times (With varying `random_state`) and average the final accuracy scores.


### Discovery of oversight and moving forward

While I was testing the models using `testModel` I discovered that the "tuned" models turned out to score a lot lower when tested multiple times over:
```
Final score for the log_regress model: 0.9441520467836256
Final score for the decision_tree model: 0.9169590643274854
```
This is something that I should have accounted for in my model tuner, I should instead have it use the `testModel` function when tuning models in order to get a more accurate score. 

I have put the old model tuner code in `tuner_old.py` and I will now be using the improved tuner code.


At this point I am basically splitting hairs with the optimization due to the small dataset. So I will keep the results for each improved test simple and just decide on a final model.

## Improved Logistic Regression Model
```
Final best params: Score: 0.9701754385964912 Params: {'exclude_variables': ('perimeter_worst', 'radius_mean', 'area_mean'), 'C': 0.9, 'test_size': 0.25, 'solver': 'liblinear', 'random_state': None}
```

## Improved Decision Tree Model
```
Final best params: Score: 0.9526315789473685 Params: {'criterion': 'log_loss', 'max_depth': 4, 'min_samples_split': 4, 'test_size': 0.25, 'exclude_variables': ['perimeter_worst', 'radius_mean', 'area_mean'], 'random_state': None}
```

## Better results with different exclude_variables
When I re-ran the logistic regression tuning (With tuning `exclude_variables`) I found a set of `exclude_variables` that seemed to improve even the decision tree model:
```
['perimeter_worst', 'radius_mean', 'area_mean']
```

# Final Results

I would have liked to test out more models, but I will have to cut it short due to time. However, both conceptually and experimentally speaking, it seems like the **Logistic Regression Model** is the best fit for this use case. 

Here are the best parameters and confusion matrix for the **Tuned Logistic Regression Model**:

#### Final Accuracy Score: 0.9701754385964912

```
{
  "exclude_variables": [
    "perimeter_worst",
    "radius_mean",
    "area_mean"
  ],
  "C": 0.9,
  "test_size": 0.25,
  "solver": "liblinear",
  "random_state": None
}
```

![Final Confusion](/images/confusion/final.png)


Overall I think that my results mostly had drawbacks due to a small dataset and not running enough iterations of tests. 