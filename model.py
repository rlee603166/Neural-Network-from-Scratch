import numpy as np

class layer:
    def __init__(self, input_layer, output_layer, activation):
        self.input = input_layer
        self.output = output_layer
        self.activation = activation
        self.w, self.b = self.initialize_parameters(input_layer, output_layer)
    
    def initialize_parameters(self, input, output):
        w = np.random.rand(output, input) * 0.01
        b = np.zeros((output, 1))
        return w, b
    
    def setLayerParams(self, w, b):
        self.w = w
        self.b = b

    def weight(self):
        return self.w
    
    def bias(self):
        return self.b

    def compute_activation(self, Z):
        if self.activation == 'relu':
            return self.relu(Z)
        if self.activation == 'sigmoid':
            return self.sigmoid(Z)
        if self.activation == 'softmax':
            return self.softmax(Z)

    # Activations
    def relu(self, Z):
        return np.maximum(0,Z)

    def sigmoid(self, Z):
        sigmoid = 1 / (1 + np.exp(-Z))
        return sigmoid
    def softmax(self, Z):
        softmax = np.exp(Z) / sum(np.exp(Z))
        return softmax

    def relu_dx(self, Z):
        return Z > 0
    
    def sigmoid_dx(self, Z):
        

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.params = {}
    
    def addLayer(self, input, output, activation):
        self.layers.append(layer(input, output, activation))
    
    def fit(X, y, learning_rate = 0.1, num_iterations= 500):
        parameters = initialize_parameters(X.shape[0])
        for i in range(num_iterations):  
            A2, cache = forward_propagation(X, parameters)
            curr_cost = compute_cost(A2,y)
            print(curr_cost)
            gradients = back_propagation(parameters, cache, X, y)
            update_parameters(parameters, gradients, learning_rate)
            if i % 100 == 0:
                costs.append(curr_cost)
                print(f'Cost for iteration {i}: {curr_cost}')
        return parameters, costs
    
    def forward_activation(self, layer, X):
        z = np.dot(layer.weight, X) + layer.bias
        a = layer.compute_activation(z)
        return z, a
            
    def forward_propagation(self, X):
        activation_layer = X
        cache = {}
        for i, layer in enumerate(self.layers):
            z, activation_layer = self.forward_activation(layer, activation_layer)
            cache['Z' + str(i+1)] = z
            cache['A' + str(i+1)] = activation_layer
        return activation_layer, cache
    

    
    def predict(X):
        ## Run forward prop and use self.parameters
        pass