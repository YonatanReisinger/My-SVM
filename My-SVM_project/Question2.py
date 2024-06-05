import pandas as pd
from SVM import SVM
from sklearn.model_selection import train_test_split
import sys
import matplotlib.pyplot as plt

def question2():
    df = pd.read_csv("simple_nonlin_classification.csv")
    true_labels = df["y"]
    feature_matrix = df.drop(columns=["y"])
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))
    draw_errors_bars(feature_matrix_train, feature_matrix_test
                     , true_labels_train, true_labels_test, C=1e20 * sys.maxsize)
    clf = SVM(kernel="rbf", C=1e20 * sys.maxsize, degree=2, gamma=2,support_vector_alpha_threshold=0.25)
    clf.fit(feature_matrix_train, true_labels_train)
    clf.draw_classification(feature_matrix_test, true_labels_test)

def draw_errors_bars(feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test, C, thresh=0.1):
    poly_dict, rbf_dict = get_erros(feature_matrix_train, feature_matrix_test
                                    , true_labels_train, true_labels_test, C, thresh)
    plot_errors(poly_dict, rbf_dict)
    plt.title("SVM Kernel Comparison")
    plt.xlabel("Kernel Parameters")
    plt.ylabel("Error")
    plt.show()

def plot_errors(poly_dict, rbf_dict):
    x_labels_poly = list(poly_dict.keys())
    y_values_poly = list(poly_dict.values())
    plt.bar(x_labels_poly, y_values_poly, color='maroon',
            width=0.4, label='Polynomial Kernel')

    x_labels_rbf = list(rbf_dict.keys())
    y_values_rbf = list(rbf_dict.values())
    # creating the bar plot
    plt.bar(x_labels_rbf, y_values_rbf, color='blue',
            width=0.4, label='RBF Kernel')

def get_erros(feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test, C, thresh=0.1):
    degrees = range(1, 4)
    gammas = range(1, 4)
    poly_dict = dict()
    rbf_dict = dict()

    for degree in degrees:
        poly_clf = SVM(kernel="polynomial", C=C, degree=degree, support_vector_alpha_threshold=thresh)
        poly_clf.fit(feature_matrix_train, true_labels_train)
        poly_dict[f"deg={degree}"] = 1 - poly_clf.score(feature_matrix_test, true_labels_test)
    for gamma in gammas:
        rbf_clf = SVM(kernel="rbf", C=C, gamma=gamma, support_vector_alpha_threshold=thresh)
        rbf_clf.fit(feature_matrix_train, true_labels_train)
        rbf_dict[f"gamma={gamma}"] = 1 - rbf_clf.score(feature_matrix_test, true_labels_test)

    return poly_dict, rbf_dict
