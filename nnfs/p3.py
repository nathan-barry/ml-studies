import numpy as np

# Make up input, weights, and biases
inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -.5, 1.0],
           [0.5, -0.91, .26, -.5],
           [-0.26, -0.27, .17, .87]]
biases = [2, 3, .5]

# The optimizer tunes the weights and biases

# Calculate outputs for each neuron
output = np.dot(weights, inputs) + biases

print(output)
