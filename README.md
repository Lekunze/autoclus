# Automated Clustering
This tool is an automated hyper-parameter search algorithm that uses meta-learning and evolutionary algorithms to find the best configurations for clustering a given dataset. Currently limited to numerical datasets, it works for all eight clustering algorithms available on [SKlearn](https://scikit-learn.org/stable/modules/clustering.html) and uses methods and parallelization from [DEAP](https://deap.readthedocs.io/en/master/index.html) to return the list of top configurations found, given parameters defined in the next section.

![Demo](https://github.com/Lekunze/autoclus/blob/master/img/app.png?raw=true)
## Parameters

This demo requires a dataset (of which samples are given), the number of generations for the evolutionary algorithm to optimize the best configurations, the population size per generation and which meta-models to be used to speed up the process. There are 2 options:
- A meta-model for recommending evaluation metrics (to optimize hyper-parameter search) and recommend clustering algorithm
- A meta-model to warm-start hyper-parameter search with a known population

The meta-features extracted from the datasets are distance-based meta-features based on this [paper](https://www.sciencedirect.com/science/article/pii/S0020025514011967).
## Setting Up

This demo requires a number of python packages that can all be installed with pip

You can ignore if you already have these packages else using pip, execute the commands:

    pip install pandas
    pip install scipy
    pip install sklearn
    pip install bokeh
    pip install deap
    pip install numpy
    pip install s_dbw
    
## Running

To run the application, navigate to the project root directory and
and execute the command:

    bokeh serve --show app.py  

The app should be running here:

    http://localhost:5006/app