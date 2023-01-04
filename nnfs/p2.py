# Make up input
inputs = [1, 2, 3, 2.5]

# Make up weights for 3 fully connected neurons (input 4 neurons, output 3 neurons)
weights1 = [0.2, 0.8, -.5, 1.0]
weights2 = [0.5, -0.91, .26, -.5]
weights3 = [-0.26, -0.27, .17, .87]

# Each neuron has a bias
bias1 = 2
bias2 = 3
bias3 = .5

# Output is sum of each input and weight multiplied added to bias
output = [
    ((inputs[0] * weights1[0])
     + (inputs[1] * weights1[1])
     + (inputs[2] * weights1[2])
     + (inputs[3] * weights1[3])
     + bias1),
    ((inputs[0] * weights2[0])
     + (inputs[1] * weights2[1])
     + (inputs[2] * weights2[2])
     + (inputs[3] * weights2[3])
     + bias2),
    ((inputs[0] * weights3[0])
     + (inputs[1] * weights3[1])
     + (inputs[2] * weights3[2])
     + (inputs[3] * weights3[3])
     + bias3),
]

print(output)
