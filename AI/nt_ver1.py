import numpy as np
import matplotlib.pyplot as plt
from keras.api.datasets import mnist

def relu(Z): #relu func for normalization at the hidden layers
    return np.maximum(0, Z)

def softmax(Z): #softmax func for normalization at the final step
    return np.exp(Z) / np.sum(np.exp(Z))

def init_params(input_size, hidden_size1, hidden_size2, output_size):
    w1 = np.random.randn(input_size, hidden_size1) * 0.01
    b1 = np.zeros((1, hidden_size1))
    w2 = np.random.randn(hidden_size1,hidden_size2) * 0.01
    b2 = np.zeros((1, hidden_size2))
    w3 = np.random.randn(hidden_size2, output_size) * 0.01
    b3 = np.zeros((1,output_size))
    return (w1,b1,w2,b2,w3,b3)

def foward_prop(input,w1, b1, w2, b2, w3, b3):
    z1 = np.dot(input, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(z1, w2) + b2
    a2 = relu(z2)
    z3 = np.dot(z2, w3) + b3
    a3 = softmax(z3)
    return (a1,a2,a3,z1,z2)

def deriv_reLu(y):
    return y > 0

def back_prop(inputs,a1,a2,w2,w3,a3,z2,z1,train):
    m = train.size
    dC3 = a3 - train
    dW3 = 1 / m * dC3.dot(a2.T)
    dB3 = 1 / m * np.sum(dC3, axis=0, keepdims=True)
    
    dC2 = w3.T.dot(dC3) * deriv_reLu(z2)
    dW2 = 1 / m * dC2.dot(a1.T)
    dB2 = 1 / m * np.sum(dC2, axis=0, keepdims=True)

    dC1 = w2.T.dot(dC2) * deriv_reLu(z1)
    dW1 = 1 / m * dC1.dot(inputs.T)
    dB1 = 1 / m * np.sum(dC1, axis=0, keepdims=True)

    return (dW3, dB3, dW2, dB2, dW1, dB1)

def update_params(w1,b1,w2,b2,w3,b3,dW3, dB3, dW2, dB2, dW1, dB1, learning_rate):
    w1 -= learning_rate * dW1
    b1 -= learning_rate * dB1
    w2 -= learning_rate * dW2
    b2 -= learning_rate * dB2
    w3 -= learning_rate * dW3
    b3 -= learning_rate * dB3
    return (w1,b2,w2,b2,w3,b3)

def predict(predictions):
    return np.argmax(predictions)

def batch_gradient_descent(train_input, train_output, iterations, learning_rate):
    w1,b1,w2,b2,w3,b3 = init_params(784, 60, 60, 10)
    for i in range(iterations):
        a1,a2,a3,z1,z2 = foward_prop(train_input,w1, b1, w2, b2, w3, b3)
        print(a3.shape)
        dW3, dB3, dW2, dB2, dW1, dB1 = back_prop(train_input, a1,a2,w2,w3,a3,z2,z1,train_output)
        w1,b2,w2,b2,w3,b3 = update_params(w1,b1,w2,b2,w3,b3,dW3,dB3,dW2,dB2,dW1,dB1, learning_rate)
        if( i % 10 == 0):
            print("Iteration", i)
            print("Accuracy: ", np.sum(predict(a3) == train_output ) / train_output.size )

    return w1,b1,w2,b2,w3,b3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 784) / 255.0  # Flatten and normalize
x_test = x_test.reshape(x_test.shape[0], 784) / 255.0
print(y_train)
y_train = np.eye(10)[y_train]  # One-hot encoding
y_test = np.eye(10)[y_test]
input_size = 784
hidden_size = 60
output_size = 10

w1, b1, w2, b2, w3, b3 = batch_gradient_descent(x_train, y_train,10, 0.01)