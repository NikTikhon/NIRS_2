from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import category_scatter, plot_confusion_matrix


def plot_features(features: np.ndarray, labels: np.ndarray):
    tsne = TSNE()
    features_2d = tsne.fit_transform(features, labels)
    plt.figure(figsize=(20, 20))
    labels = np.expand_dims(labels, axis=1)
    features_2d = np.concatenate((labels.T, features_2d.T))
    features_2d = features_2d.T
    fig = category_scatter(x=1, y=2, label_col=0, data=features_2d, markersize = 10 )
    # plt.scatter(features_2d[:, 0], features_2d[:, 1])
    plt.show()


def plot_conf_m(conf_mat: np.ndarray, title: str):
    plot_confusion_matrix(conf_mat)
    plt.show()


def main():
    path = r'E:\Programs\PycharmProjects\NIRS_2\visual\samy_set_v3_WithRot_LR0001_b128.csv'
    df = pd.read_csv(path)
    X = df.loc[:, ~df.columns.isin(['labels', 'audio_names'])]
    y = df['labels']
    X = X.values
    y = y.values
    plot_features(X, y)
    pass

if __name__ == '__main__':
    main()