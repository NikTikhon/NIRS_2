import pandas as pd
import joblib
from classification.random_forest import optimize_hyperparams, classify


# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer='../data/features/samy_set_v3_WithRot_LR0001_b128.csv')
X = df.loc[:, ~df.columns.isin(['labels', 'audio_names'])]
y = df['labels']

img_ids = df['audio_names']

print('Has NaN:', df.isnull().values.any())

hyper_params = [{'max_depth': [5, 20, 50, 100, 150, 200], 'n_estimators': [10, 50, 100, 150, 200], 'criterion':['gini', 'entropy']}]
# hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5, 1e-6], 'C': [1, 10, 100]}]

classifier_file_path = '../data/svm_classifier/samy_set_v3_SVM_rbf_C_1_gamma_0.01.pkl'

best_estimator, opt_params = optimize_hyperparams(X, y, params=hyper_params)
joblib.dump(best_estimator, classifier_file_path)

classify(X, y, opt_params)
# print_confusion_matrix(X, y, opt_params)








