from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import seaborn as sn


def optimize_hyperparams(x, y, params):
    """
    Hyperparameter optimization of the SVM
    :param x: The feature vectors
    :param y: The labels
    :param params: The grid of all the possible optimal hyperparameters
    :returns: The optimal hyperparameters
    """
    # Optimize hyper-parameters
    model = LogisticRegression()
    model_grid_search = GridSearchCV(model, params, cv=10, n_jobs=5)
    model_grid_search.fit(x.values, y.values)
    print("Optimal hyper-parameters: ", model_grid_search.best_params_)
    print("Accuracy :", model_grid_search.best_score_)
    return model_grid_search.best_estimator_, model_grid_search.best_params_


def classify(x, y, opt_params):
    """
    Classify a feature vector using SVM and print some metrics
    :param x: The feature vectors
    :param y: The labels
    :param opt_params: The optimal hyperparameters
    """
    # Single SVM run with optimized hyperparameters and
    model = LogisticRegression(C=opt_params['C'], solver=opt_params['solver'])
    scores = cross_val_score(model, x, y, cv=10, scoring='accuracy', n_jobs=-1)
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))