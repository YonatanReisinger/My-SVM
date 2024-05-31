import numpy as np
import math

def poly_kernel(X, y, degree: int):
    return (1 + X.T @ y) ** degree

def RBF_kernel(X, y, gamma: float):
    return np.e ** (-gamma ((X - y).T @ (X - y)))

def sigmoid_kernel(X, y, gamma: float, r: float):
    z = gamma * X.T @ y + r
    return math.tanh(z)