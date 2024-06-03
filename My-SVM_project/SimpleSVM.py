import pandas as pd
import numpy as np
import scipy.sparse as sp
import qpsolvers as qps
import osqp
import matplotlib.pyplot as plt

class SimpleSVM:
    def __init__(self, fit_intercept=True, optimization_form = "primal", zero_thresh = 0.1):
        self.__weights = None
        # The model expect that the true labels will be 1 and -1
        self.__positive_label = 1
        self.__negative_label = -1
        self.__fit_intercept = fit_intercept
        self.set_optimization_form(optimization_form)
        self.__fit_completed = False
        self.__margin = 1
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = None
        self.__y_train_np = None
        self.__original_labels = tuple()
        self.__support_vectors = None
        self.__zero_thresh = zero_thresh

    def __set_fit_params(self, X, y):
        unique_labels = np.unique(y)

        if len(unique_labels) != 2:
            raise RuntimeError("Model is binary. can train just on classifications with just 2 classes")

        if self.__fit_intercept:
            X = SimpleSVM.add_intercept(X)

        self.__original_labels = unique_labels
        y = np.where(y == unique_labels[0], self.__negative_label, self.__positive_label)
        # Save the data as numpy matrix in order to make calculations
        self.__X_train_np = np.array(X)
        self.__y_train_np = np.array(y)

    def fit(self, X, y):
        self.__set_fit_params(X, y)
        if self.__optimization_form == "primal":
            self.__fit_primal()
            self.__fit_completed = True
        elif self.__optimization_form == "dual":
            self.__fit_dual()
            self.__fit_completed = True

    def __fit_primal(self):
        X = self.__X_train_np
        y = self.__y_train_np

        num_of_samples, num_of_features = X.shape
        P = sp.eye(num_of_features, format='csc')
        q = np.zeros(num_of_features)
        G = -sp.csc_matrix(np.diag(y)) @ sp.csc_matrix(X)
        h = -np.ones(num_of_samples)

        self.__weights = qps.solve_qp(P, q, G, h, solver="osqp")

    def __fit_dual(self):
        X = self.__X_train_np
        y = self.__y_train_np

        num_of_samples = X.shape[0]
        G = sp.csc_matrix(np.diag(y) @ X)
        P = G @ G.T
        q = -np.ones(num_of_samples)
        GG = -sp.eye(num_of_samples, format='csc')
        h = np.zeros(num_of_samples)
        alpha = qps.solve_qp(P, q, GG, h, solver='osqp')
        support_vectors_indices = np.argwhere(np.abs(alpha) > self.__zero_thresh).reshape(-1)
        self.__support_vectors = X[support_vectors_indices]
        self.__weights = G.T @ alpha


    def predict(self, X):
        if self.__fit_completed:
            feature_matrix = np.array(X)
            if self.__fit_intercept:
                feature_matrix = SimpleSVM.add_intercept(feature_matrix)
            predictions = np.apply_along_axis(self.predict_label_for_single_feature_vector
                                            , axis=1, arr=feature_matrix).tolist()
            return predictions
        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def predict_label_for_single_feature_vector(self, feature_vector):
        if np.dot(self.__weights, feature_vector) > 0:
            return self.__original_labels[1]
        else:
            return self.__original_labels[0]

    def score(self, X, y):
        if self.__fit_completed:
            predictions = np.array(self.predict(X))
            true_labels = np.array(y)
            num_of_correct_classifications = np.sum(predictions == true_labels)
            return num_of_correct_classifications / len(y)

        else:
            raise RuntimeError("Model has not been fitted yet. Please call fit() first.")

    def draw_classification(self, X, y):
        if X.shape[1] == 2:
            X = np.array(X)
            y = np.array(y)
            x_min = X.min(axis=0)[0]
            x_max = X.max(axis=0)[0]
            y_min = X.min(axis=0)[1]
            y_max = X.max(axis=0)[1]

            # Draw data points
            red = np.where(y <= 0)
            blue = np.where(y > 0)
            plt.plot(X[red, 0], X[red, 1], 'o', color='red')
            plt.plot(X[blue, 0], X[blue, 1], 'o', color='blue')

            # Draw support vectors from the fit process
            if self.__support_vectors is not None:
                plt.plot(self.__support_vectors[:, 0], self.__support_vectors[:, 1], 'o', color='green')
                plt.scatter(self.__support_vectors[:, 0], self.__support_vectors[:, 1], s=100, facecolors='none',
                            edgecolors='k', label='Support Vectors')

            self.__draw_hyperplane(x_min, x_max)

            plt.axis([x_min - 1, x_max + 1, y_min - 1, y_max + 1])
            plt.show()
        else:
            raise ValueError("just 2D plotting is supported")

    def __draw_hyperplane(self, x_min, x_max):
        if self.__weights is not None:
            w0, w1, bias = self.__weights
            num_of_points_on_line = 1000
            hyperplane_x_values = np.linspace(x_min, x_max, num_of_points_on_line)
            # Calculate corresponding y values using the equation
            hyperplane_y_values = (-w0 * hyperplane_x_values - bias) / w1
            # Plot the line
            plt.plot(hyperplane_x_values, hyperplane_y_values, color="black")
            self.__draw_margin_lines(x_min, x_max)
        else:
            raise ValueError("weights were not found, fit was not called or a problem occurred in the fit process")

    def __draw_margin_lines(self, x_min, x_max):
        w0, w1, bias = self.__weights
        num_of_points_on_line = 1000
        margin_line_x_values = np.linspace(x_min, x_max, num_of_points_on_line)
        # Calculate corresponding y values using the equation
        margin_line_y_values1 = (self.__margin -w0 * margin_line_x_values - bias) / w1
        margin_line_y_values2 = (- self.__margin -w0 * margin_line_x_values - bias) / w1
        # Plot the first part of the margin
        plt.plot(margin_line_x_values, margin_line_y_values1, color="blue", linestyle=":")
        plt.plot(margin_line_x_values, margin_line_y_values2, color="red", linestyle=":")


    def __calc_distance_from_hyperplane(self, feature_vector):
        return abs(np.dot(self.__weights, feature_vector)) / np.linalg.norm(self.__weights)

    # ------------------- Getters -------------------
    def get_weights(self):
        return self.__weights

    # ------------------- Setters -------------------
    def set_optimization_form(self, optimization_form):
        valid_optimization_forms = ["primal", "dual"]
        if optimization_form in valid_optimization_forms:
            self.__optimization_form = optimization_form

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