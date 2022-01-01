from sklearn import svm
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
    # model = svm.SVC(probability=True)
    model = neighbors.KNeighborsClassifier()
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
    model = neighbors.KNeighborsClassifier(n_neighbors=opt_params['n_neighbors'], weights=opt_params['weights'], leaf_size = opt_params['leaf_size'])
    scores = cross_val_score(model, x, y, cv=10, scoring='accuracy', n_jobs=-1)
    print(scores)
    print(np.mean(scores))
    print(np.std(scores))


# def print_confusion_matrix(x, y, opt_params):
#     """
#     Print the confusion matrix of an SVM classification
#     :param x: The feature vectors
#     :param y: The labels
#     :param opt_params: The optimal hyperparameters
#     """
#     y_pred, y_test = get_predictions(x, y, opt_params)
#     # Printing out false/true positives/negatives
#     tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
#     print('True negatives: ', tn, 'False positives: ', fp, 'False negatives: ', fn, 'True positives: ', tp)
#
#     # Using seaborn to create a confusion matrix table
#     data = {'y_Predicted': y_pred, 'y_Actual': y_test}
#     df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
#     conf_matrix = pd.crosstab(df['y_Actual'], df['y_Predicted'], rownames=['Actual'], colnames=['Predicted'])
#     sn.heatmap(conf_matrix, cmap=ListedColormap(['#ED7D31', '#009FDA']), annot=True, fmt='g', cbar=False)




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
    model = neighbors.KNeighborsClassifier(n_neighbors=opt_params['n_neighbors'], weights=opt_params['weights'], leaf_size = opt_params['leaf_size'])
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    return y_pred, y_test
