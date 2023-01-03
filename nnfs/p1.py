# Make up input and weights
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]

# Each neuron has a bias
bias = 3

# Output is sum of each input and weight multiplied added to bias
output = ((inputs[0] * weights[0])
          + (inputs[1] * weights[1])
          + (inputs[2] * weights[2])
          + bias)

print(output)
