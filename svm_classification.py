import pandas as pd
import joblib
from classification.SVM import optimize_hyperparams, classify, print_confusion_matrix


# Read features and labels from CSV
df = pd.read_csv(filepath_or_buffer='data/features/samy_set_v3_WithRot_LR0001_b128.csv')
X = df.loc[:, ~df.columns.isin(['labels', 'audio_names'])]
y = df['labels']

img_ids = df['audio_names']

print('Has NaN:', df.isnull().values.any())

hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}]
#hyper_params = [{'kernel': ['rbf'], 'gamma': [1e-2], 'C': [0.01]}]

classifier_file_path = 'data/svm_classifier/samy_set_v3_SVM_rbf_C_1_gamma_0.01.pkl'

best_estimator, opt_params = optimize_hyperparams(X, y, params=hyper_params)
joblib.dump(best_estimator, classifier_file_path)

classify(X, y, opt_params)
print_confusion_matrix(X, y, opt_params)








