import pandas as pd
import numpy as np
import scipy.sparse as sp
import qpsolvers as qps
import osqp
import matplotlib.pyplot as plt
import Kernel
import itertools

class SVM:
    def __init__(self, kernel="rbf", degree=3, C=1, gamma=1, r=10
                 , fit_intercept=True, support_vector_alpha_threshold=0.1):
        self.__kernel = kernel
        self.__degree = degree
        self.__C = C
        self.__gamma = gamma
        self.__r = r
        self.__init_kernel_func()
        # The model expect that the true labels will be 1 and -1
        self.__positive_label = 1
        self.__negative_label = -1
        self.__fit_intercept = fit_intercept
        self.__fit_completed = False
        self.__margin = 1
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = None
        self.__y_train_np = None
        self.__original_labels = tuple()
        self.__alphas = None
        self.__support_vectors_indices = None
        self.__support_vectors = None
        self.__support_vectors_true_labels = None
        self.__support_vector_alpha_threshold = support_vector_alpha_threshold

    def __init_kernel_func(self):
        match self.__kernel:
            case "linear":
                self.__kernel_func = Kernel.linear_kernel
            case "rbf":
                self.__kernel_func = Kernel.RBF_kernel_generator(self.__gamma)
            case "polynomial":
                self.__kernel_func = Kernel.poly_kernel_generator(self.__degree)
            case "sigmoid":
                self.__kernel_func = Kernel.sigmoid_kernel_generator(self.__gamma, self.__r)
            case other:
                raise ValueError(f"{self.__kernel} does not exist")

    def __set_fit_params(self, X, y):
        unique_labels = np.unique(y)

        if len(unique_labels) != 2:
            raise RuntimeError("Model is binary. can train just on classifications with just 2 classes")

        if self.__fit_intercept:
            X = SVM.add_intercept(X)

        self.__original_labels = unique_labels
        y = np.where(y == unique_labels[0], self.__negative_label, self.__positive_label)
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = np.array(X)
        self.__y_train_np = np.array(y)

    def fit(self, X, y):
        self.__set_fit_params(X, y)
        self.__fit_dual()
        self.__fit_completed = True

    def __fit_dual(self):
        X = self.__X_train_np
        y = self.__y_train_np

        num_of_samples = X.shape[0]
        K = self.__compute_kernel_matrix(X)
        P = sp.csc_matrix(np.outer(y, y) * K)
        P = 0.5 * (P + P.T)
        P += np.eye(num_of_samples)
        P = sp.csc_matrix(P)
        q = -np.ones(num_of_samples)
        G = sp.csc_matrix(np.block([[-np.eye(num_of_samples)], [np.eye(num_of_samples)]]))
        h = np.block([np.zeros(num_of_samples), self.__C * np.ones(num_of_samples)])

        self.__alphas = qps.solve_qp(P, q, G, h, solver='osqp')
        self.__find_support_vectors_indices()
        self.__find_support_vectors()
        self.__find_support_vectors_alphas()

    def __compute_kernel_matrix(self, X):
        num_of_samples = X.shape[0]
        K = np.zeros((num_of_samples, num_of_samples))
        for i in range(num_of_samples):
            for j in range(num_of_samples):
                K[i, j] = self.__kernel_func(X[i, :], X[j, :])
        return K

    # TODO:
    def __find_matrices_for_optimzation(self):
        pass

    def __find_support_vectors_indices(self):
        self.__support_vectors_indices = np.argwhere(np.abs(self.__alphas) > self.__support_vector_alpha_threshold).reshape(-1)

    def __find_support_vectors(self):
        self.__support_vectors = self.__X_train_np[self.__support_vectors_indices]
        self.__support_vectors_true_labels = self.__y_train_np[self.__support_vectors_indices]

    def __find_support_vectors_alphas(self):
        self.__alphas = self.__alphas[self.__support_vectors_indices]


    def predict(self, X):
        if self.__fit_completed:
            feature_matrix = np.array(X)
            if self.__fit_intercept:
                feature_matrix = SVM.add_intercept(feature_matrix)
            predictions_decision_values = np.array(self.decision_function(feature_matrix))
            predictions = np.where(predictions_decision_values > 0
                                   , self.__original_labels[1], self.__original_labels[0])
            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def decision_function(self, X) -> np.array:
        ROWS = 1
        decision_values = np.apply_along_axis(self.__get_decision_val_for_single_feature_vector, arr=X, axis=ROWS)

        return decision_values

    def __get_decision_val_for_single_feature_vector(self, new_feature_vector):
        ROWS = 1
        kernel_values = np.apply_along_axis(self.__kernel_func, ROWS,
                                            self.__support_vectors, new_feature_vector)
        decision_value = np.dot(self.__alphas * self.__support_vectors_true_labels, kernel_values)

        return decision_value

    def score(self, X, y):
        if self.__fit_completed:
            predictions = np.array(self.predict(X))
            true_labels = np.array(y)
            num_of_correct_classifications = np.sum(predictions == true_labels)
            return num_of_correct_classifications / len(y)
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def draw_classification(self, X, y):
        if X.shape[1] != 2:
            raise ValueError("just 2D plotting is supported")
        elif not self.__fit_completed:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")
        # else
        X = np.array(X)
        y = np.array(y)
        self.__draw_data_points(X, y)
        self.__draw_support_vectors()
        self.__draw_hyperplane(X, y)
        plt.show()


    def __draw_data_points(self, X, y):
        red = np.where(y == self.__original_labels[0])
        blue = np.where(y == self.__original_labels[1])
        plt.plot(X[red, 0], X[red, 1], 'o', color='red')
        plt.plot(X[blue, 0], X[blue, 1], 'o', color='blue')

    def __draw_support_vectors(self):
        if self.__support_vectors is not None:
            plt.plot(self.__support_vectors[:, 0], self.__support_vectors[:, 1], 'o', color='green')
            plt.scatter(self.__support_vectors[:, 0], self.__support_vectors[:, 1], s=100, facecolors='none',
                        edgecolors='k', label='Support Vectors')

    def __draw_hyperplane(self, X, y):
        x_min = X.min(axis=0)[0]
        x_max = X.max(axis=0)[0]
        y_min = X.min(axis=0)[1]
        y_max = X.max(axis=0)[1]

        xx = np.linspace(x_min, x_max)
        yy = np.linspace(y_min, y_max)

        xx, yy = np.meshgrid(xx, yy)

        N = X.shape[0]
        z = np.zeros(xx.shape)
        for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
            z[i, j] = sum([y[k] * self.__alphas[k] * self.__kernel_func(X[k, :], np.array([xx[i, j], yy[i, j]])) for k in range(N)])

        plt.rcParams["figure.figsize"] = [15, 10]
        plt.contour(xx, yy, z, levels=[-self.__margin, 0, self.__margin]
                    , colors=['red', 'black', 'blue'], linestyles=['--', '-', '--'])

    # ------------------- static methods -------------------
    @staticmethod
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