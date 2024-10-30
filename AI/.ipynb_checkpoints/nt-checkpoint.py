import numpy as np
import matplotlib.pyplot as plt
from kernas.datasets import mnist

def relu(Z): #relu func for normalization at the hidden layers
    return np.maximum(0, Z)

def softmax(Z): #softmax func for normalization at the final step
    expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return expZ / np.sum(expZ, axis=1, keepdims=True)

def foward_prop(input,h1_weights, h1_bias, h2_weights, h2_bias, final_weights, final_bias):
    h1 = np.dot(input, h1_weights) + h1_bias
    h1 = relu(h1)
    h2 = np.dot(h1, h2_weights) + h2_bias
    h2 = relu(h2)
    final = np.dot(h2, final_weights) + final_bias
    final = softmax(final)
    return (h1,h2,final)

def back_prop(input,h1_weights, h1_bias, h2_weights, h2_bias, final_weights, final_bias):


def init_params(input_size, hidden_size1, hidden_size2, output_size):
    h1_weights = np.random.randn(input_size, hidden_size1) * 0.01
    h1_bias = np.zeros(1, hidden_size1)
    h2_weights = np.random.randn(hidden_size1,hidden_size2) * 0.01
    h2_bias = np.zeros(1, hidden_size2)
    final_weights = np.random.randn(hidden_size2, output_size) * 0.01
    final_bias = np.zeros(1,output_size)
    return (h1_weights, h1_bias, h2_weights, h2_bias, final_weights, final_bias)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 784) / 255.0  # Flatten and normalize
    X_test = X_test.reshape(X_test.shape[0], 784) / 255.0
    X_test = X_test.reshape(X_test.shape[0], 784) / 255.0
    y_train = np.eye(10)[y_train]  # One-hot encoding
    y_test = np.eye(10)[y_test]
    input_size = 784
    hidden_size = 64
    output_size = 10