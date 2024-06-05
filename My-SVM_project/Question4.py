import pandas as pd
from sklearn.model_selection import train_test_split
import sys
from Question2 import draw_errors_bars
from sklearn.decomposition import PCA

def question4():
    df = pd.read_csv("Processed Wisconsin Diagnostic Breast Cancer.csv")
    true_labels = df["diagnosis"]
    feature_matrix = df.drop(columns=["diagnosis"])
    pca = PCA(n_components=12)
    feature_matrix = pca.fit_transform(feature_matrix)
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))
    draw_errors_bars(feature_matrix_train, feature_matrix_test
                     , true_labels_train, true_labels_test, C=sys.maxsize/100, thresh=1e-16)

