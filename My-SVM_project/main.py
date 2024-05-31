import pandas as pd
from SVM import SVM
from sklearn.model_selection import train_test_split
import sys

def question1():
    df = pd.read_csv("simple_classification.csv")
    true_labels = df["y"]
    feature_matrix = df.drop(columns=["y"])
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))
    question1_primal(feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test)
    question1_dualic(feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test)

def question1_primal(feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test):
    clf = SVM(optimization_form="primal", C=sys.maxsize)
    clf.fit(feature_matrix_train, true_labels_train)
    print(f"The weights using primal fit are: {clf.get_weights()}")
    clf.draw_classification(feature_matrix_test, true_labels_test)

def question1_dualic(feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test):
    clf = SVM(optimization_form="dual", C=sys.maxsize)
    clf.fit(feature_matrix_train, true_labels_train)
    print(f"The weights using dual fit are: {clf.get_weights()}")
    clf.draw_classification(feature_matrix_test, true_labels_test)

def question2():
    df = pd.read_csv("simple_nonlin_classification.csv")
    true_labels = df["y"]
    feature_matrix = df.drop(columns=["y"])
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))
    clf = SVM(optimization_form="primal")
    clf.fit(feature_matrix_train, true_labels_train)
    print(f"The weights using primal fit are: {clf.get_weights()}")
    clf.draw_classification(feature_matrix_test, true_labels_test)


if __name__ == '__main__':
    question1()
