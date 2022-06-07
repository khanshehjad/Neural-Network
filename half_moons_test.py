from model import NeuralNet
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets

# Generate a dataset and plot it
np.random.seed(0)
X, Y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:, 0], X[:, 1], s=40, c=Y, cmap=plt.cm.Spectral)
# plt.savefig("data.png")

# Build model with 3 hidden layers
model = NeuralNet([2, 3, 2], activation_function='tanh', print_cost=True)
model.build_model(X, Y)

# Plot decision boundary
