#!/bin/python3

# This is the old version of my "Model Tuner", more about it is explained in the "Discovery of oversight" section in README.md

# This can be universally used to mass-test hyperparameters to optimize models via the `tune` function
# The time this can take will increase by O(2^n) (I think). Where n is the number of arguments tested
# Re-writing this in rust or something would make it much faster

# This uses multiprocessing to speed things up

# `model_function` - The test function for the model, must return a tuple of (confusion_matrix, accuracy_score)
# `dataset` - The pandas dataset you will be using
# `important_variables` - The columns in the dataset that will be used as dependant variables
# `params` - The paramaters to test with in the model_function
# params Example for the LogisticRegression model:
# {
#   "random_state": 16, # The random_state is fixed for every test
#   "test_size": {"type": float, "min": 0.05, "max": 0.95, "interval": 0.5},
#   "solver": {"type": str, "vals": ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]},
#   "exclude_variables": {"type": list, "len": 1, "vals": ["area_mean", "area_se", "area_worst"]}
# }
#  
# If a param `type`
# This function will return the paramaters that give the maximum accuracy score

# TODO: MultiProcessing

from itertools import product, combinations
import decimal
import math
import multiprocessing

# Shamelessly copied from stackoverflow
def drange(x, y, jump):
    x = decimal.Decimal(x)
    y = decimal.Decimal(y)
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)

# pqueue is used to put the result
def runTests(model_function, dataset, fixedParams, allTuneParams, pqueue):
    best_params_score = 0
    best_params = {}
    for tuneParams in allTuneParams:
        tuneParams.update(fixedParams) # Add the fixed params
        cnf_matrix, score = model_function(dataset, **tuneParams)
        if score > best_params_score:
            best_params = tuneParams
            best_params_score = score
            print('New best params found! Score: %s Params: %s' % (score, tuneParams))
    pqueue.put((best_params_score,best_params))

def tune(model_function, dataset, important_variables, params):
    fixed_params = {}
    tune_params = {}
    # Seperated fixed params from the ones we are going to text
    for param in params:
        if type(params[param]) == dict:
            tune_params[param] = params[param]
        else:
            fixed_params[param] = params[param]

    # In here we nest lists of all the possible combinations of param values
    # We then take the itertools product of this to get all our possible parameters
    tuneParams = []  
    
    for tuneParam in tune_params:
        # See what type of parameter it is, and get all the possible values
        if tune_params[tuneParam]['type'] == float:
            tuneParams.append(list(drange(tune_params[tuneParam]['min'],tune_params[tuneParam]['max'],tune_params[tuneParam]['interval'])))
        elif tune_params[tuneParam]['type'] == int:
            tuneParams.append(list(range(tune_params[tuneParam]['min'],tune_params[tuneParam]['max'],tune_params[tuneParam]['interval'])))
        elif tune_params[tuneParam]['type'] == str:
            tuneParams.append(tune_params[tuneParam]['vals'])
        elif tune_params[tuneParam]['type'] == list:
            # Get all combinations of specified size
            tuneParams.append(list(combinations(tune_params[tuneParam]['vals'],tune_params[tuneParam]['len'])))
        else:
            raise BaseException('Unknown param type supplied: %s' % (str(tune_params[tuneParam]['type'])))
    
    # All the possible configurations of tuneParams
    tuneParams = list(product(*tuneParams))
    # Reassign param names (Convert them back to dicts)
    allTuneParams = []
    for params in tuneParams:
        newParams = {}
        i = 0
        for tuneParam in tune_params:
            newParams[tuneParam] = params[i]
            i += 1
        allTuneParams.append(newParams)

    print("Model Tests To Run: %s" % (len(allTuneParams)))
    cores = multiprocessing.cpu_count()
    # Distribute the work over all CPU cores
    print("CPU Cores Detected: %s" % (cores))
    chunkSize = len(allTuneParams)/cores
    multiprocessing.set_start_method('spawn')
    queue = multiprocessing.Queue()
    for i in range(0,cores):
        chunk = allTuneParams[math.floor(i*chunkSize):math.floor((i+1)*(chunkSize))]
        if i == cores-1:
            chunk = allTuneParams[math.floor(i*chunkSize):] # Distrubute the leftovers to the last core
        p = multiprocessing.Process(target=runTests, args=(model_function,dataset,fixed_params,chunk,queue,))
        p.start()

    best_params_score = 0
    best_params = {}
    
    # Wait for all the results to come in
    for i in range(0,cores):
        score, params = queue.get()
        if score > best_params_score:
            best_params_score = score
            best_params = params
        print('One chunk has finished')

    print('All done!')
    print('Final best params: Score: %s Params: %s' % (best_params_score, best_params))