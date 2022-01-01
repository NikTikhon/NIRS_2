from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
import seaborn as sn
from xgboost.sklearn import XGBClassifier

def optimize_hyperparams(x, y, params):
    """
    Hyperparameter optimization of the SVM
    :param x: The feature vectors
    :param y: The labels
    :param params: The grid of all the possible optimal hyperparameters
    :returns: The optimal hyperparameters
    """
    # Optimize hyper-parameters
    model = XGBClassifier()
    model_grid_search = GridSearchCV(model, params, cv=10, n_jobs=3)
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
    model = XGBClassifier(n_estimators=opt_params['n_estimators'], verbosity=0, booster=opt_params['booster'])
    scores = cross_val_score(model, x, y, cv=10, scoring='accuracy', n_jobs=3)
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))




def get_predictions(x, y, opt_params):
    """
    Classification using SVM
    :param x: The feature vectors
    :param y: The labels
    :param opt_params: The optimal hyperparameters
    :returns: The predicted and true labels
    """
    # Run one SVM with 80-20 split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)
    model = svm.SVC(kernel='rbf', gamma=opt_params['gamma'], C=opt_params['C'])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_pred, y_test
