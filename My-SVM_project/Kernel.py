import numpy as np

def linear_kernel(x, y):
    return np.dot(x, y)

def poly_kernel_generator(degree: int):
    def poly_kernel(x, y):
        return (1 + np.dot(x, y)) ** degree
    return poly_kernel

def RBF_kernel_generator(gamma: float):
    def RBF_kernel(x, y):
        return np.e ** (-gamma * np.linalg.norm(x - y) ** 2)
    return RBF_kernel

def sigmoid_kernel_generator(gamma: float, r: float):
    def sigmoid_kernel(x, y):
        z = np.dot(x, y) * gamma + r
        return np.tanh(z)
    return sigmoid_kernel