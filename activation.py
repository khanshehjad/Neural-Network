import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))

    assert(A.shape == Z.shape)
    return A


def derivative_sigmoid(Z):
    dZ = sigmoid(Z) * (1 - sigmoid(Z))

    assert(dZ.shape == Z.shape)
    return dZ

def relu(Z):
    A = np.maximum(0, Z)

    assert(A.shape == Z.shape)
    return A

def derivative_relu(Z):
    dZ = np.array([1 if i else 0 for i in range(Z)])

    assert(dZ.shape == Z.shape)
    return dZ

def tanh(Z):
    A = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))

    assert(A.shape == Z.shape)
    return A

def derivative_tanh(Z):
    dZ = 1 - tanh(Z) * tanh(Z)

    assert(dZ.shape == Z.shape)
    return dZ
