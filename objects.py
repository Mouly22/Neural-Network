import numpy as np
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        
layer1 = Layer_Dense(4, 5)
layer2 = Layer_Dense(5, 2)

layer1.forward(X)
#print(layer1.output)
#Output for layer1: [because X is 3x4 and layer 1 is 4x5; output1 is 3x5]
#[[-0.11423289  0.24392004 -0.55421739 -0.0876068   0.51671213]
# [ 0.45387967  1.16119616 -0.1350225   0.35772809 -0.42846762]
# [-0.75536804 -0.88741844 -0.32012558 -0.03810369  1.17626654]]
layer2.forward(layer1.output)
print(layer2.output) #[output1 is 3x5 and layer2 is 5x2; final output is 3x2]
#final output of layer2:
#[[0.22808272 0.2194206 ]
# [0.0719297  0.24311985]
# [0.14012915 0.01086533]]
