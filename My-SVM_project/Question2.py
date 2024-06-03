import pandas as pd
from SVM import SVM
from sklearn.model_selection import train_test_split

def question2():
    df = pd.read_csv("simple_nonlin_classification.csv")
    true_labels = df["y"]
    feature_matrix = df.drop(columns=["y"])
    feature_matrix_train, feature_matrix_test, true_labels_train, true_labels_test = (
        train_test_split(feature_matrix, true_labels, test_size=0.2, shuffle=True))
    clf = SVM(optimization_form="dual", C=sys.maxsize, kernel="polynomial", degree=2)
    clf.fit(feature_matrix_train, true_labels_train)
    print(f"The weights using primal fit are: {clf.get_weights()}")
    print(clf.score(feature_matrix_test, true_labels_test))
    clf.draw_classification(feature_matrix_test, true_labels_test)