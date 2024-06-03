import numpy as np
import math


def linear_kernel(first_vector, second_vector):
    return np.dot(first_vector, second_vector)

def poly_kernel_generator(degree: int):
    def poly_kernel(first_vector, second_vector):
        return (1 + np.dot(first_vector, second_vector)) ** degree
    return poly_kernel

def RBF_kernel_generator(gamma: float):
    def RBF_kernel(first_vector, second_vector):
        return np.e ** (-gamma * ((first_vector - second_vector).T @ (first_vector - second_vector)))
    return RBF_kernel

def sigmoid_kernel_generator(gamma: float, r: float):
    def sigmoid_kernel(first_vector, second_vector):
        z = gamma * first_vector.T @ second_vector + r
        return math.tanh(z)
    return sigmoid_kernel