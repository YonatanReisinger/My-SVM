import numpy as np
import scipy.sparse as sp
import pandas as pd
import qpsolvers as qps
import osqp
import matplotlib.pyplot as plt
from SVM import SVM

def question1():
    df = pd.read_csv("simple_classification.csv")
    true_labels = df["y"]
    feature_matrix = df.drop(columns=["y"])
    feature_matrix = feature_matrix.to_numpy()
    true_labels = true_labels.to_numpy()
    question1_quadratic(feature_matrix, true_labels)

def question1_quadratic(feature_matrix, true_labels):
    svm_model = SVM()
    svm_model.fit_primal(feature_matrix, true_labels)
    svm_model.draw_classification(feature_matrix, true_labels)

def draw_simple_classification(feature_matrix: np.array, true_labels: np.array, svm_model: SVM):
    x_min = feature_matrix.min(axis=0)[0]
    x_max = feature_matrix.max(axis=0)[0]
    y_min = feature_matrix.min(axis=0)[1]
    y_max = feature_matrix.max(axis=0)[1]

    draw_data_points(feature_matrix, true_labels)
    svm_model.draw_hyperplane(x_min, x_max)

    plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])
    plt.show()

def draw_data_points(feature_matrix, true_labels):
    red = np.where(true_labels <= 0)
    blue = np.where(true_labels > 0)
    plt.plot(feature_matrix[red, 0], feature_matrix[red, 1], 'o', color='red')
    plt.plot(feature_matrix[blue, 0], feature_matrix[blue, 1], 'o', color='blue')


def question1_dualic(df):
    pass

def add_intercept(feature_matrix):
    # Add 1 to every feature vector
    if isinstance(feature_matrix, np.ndarray):
        intercept_column = np.ones((feature_matrix.shape[0], 1))
        feature_matrix_with_constant = np.concatenate((feature_matrix, intercept_column), axis=1)
        return np.array(feature_matrix_with_constant)  # Returning the modified feature_matrix
    elif isinstance(feature_matrix, pd.DataFrame):
        feature_matrix_with_constant = feature_matrix.copy()
        feature_matrix_with_constant["constant"] = 1
        return pd.DataFrame(feature_matrix_with_constant)  # Returning the modified DataFrame


if __name__ == '__main__':
    question1()
