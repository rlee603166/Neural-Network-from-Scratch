import numpy as np

class NeuralNetwork:
    def __init__(self):
        self.parameters = {}
        self.gradients = {}
        self.cache = {}
        self.learning_rate = 0
        self.seed = np.random.seed(42)
    
    def label_encoder(self, y):
        zeros = np.zeros((y.shape[0], np.max(y)+1))
        for i, num in enumerate(y):
            zeros[i][np.squeeze(num)] = 1
        return zeros.T

    def relu(self, Z):
        return np.maximum(0,Z)
    
    def sigmoid(self, Z):
        sigmoid = 1 / (1 + np.exp(-Z))
        return sigmoid
    
    def softmax(self, Z):
        softmax = np.exp(Z) / sum(np.exp(Z))
        return softmax
    
    def relu_deriv(self, Z):
        return Z > 0

    def initialize_parameters(self, n_x, n_h=10, n_y=10):
        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        parameters = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }
        return parameters
    
    # Forward prop
    def forward_propagation(self, X):
        W1, b1, W2, b2 = self.parameters.values()

        Z1 = np.dot(W1, X) + b1
        A1 = self.relu(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.softmax(Z2)
        self.cache = {
            'Z1': Z1,
            'A1': A1,
            'Z2': Z2,
            'A2': A2
        }

        return A2
        
    # Back prop method 
    def back_propagation(self, X, y):
        W1, b1, W2, b2 = self.parameters.values()
        Z1, A1, Z2, A2 = self.cache.values()
    
        m = X.shape[1]
        y = self.label_encoder(y)
    
        dZ2 = A2 - y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * self.relu_deriv(Z1)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.gradients = {
            'dZ2': dZ2,
            'dW2': dW2,
            'db2': db2,
            'dZ1': dZ1,
            'dW1': dW1,
            'db1': db1
        }
    
    def update_parameters(self):
        W1 = self.parameters['W1'].copy()
        b1 = self.parameters['b1'].copy()
        W2 = self.parameters['W2'].copy()
        b2 = self.parameters['b2'].copy()
        
        self.parameters['W1'] = W1 - self.learning_rate * self.gradients['dW1']
        self.parameters['b1'] = b1 - self.learning_rate * self.gradients['db1']
        self.parameters['W2'] = W2 - self.learning_rate * self.gradients['dW2']
        self.parameters['b2'] = b2 - self.learning_rate * self.gradients['db2']

    
    # Methods for results
    def get_predictions(self, A2):
        return np.argmax(A2,0)

    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size
    
    def fit(self, X, y, learning_rate = 0.1, num_iterations = 1000):
        self.learning_rate = learning_rate
        self.parameters = self.initialize_parameters(X.shape[0])
        for i in range(num_iterations):  
            A2 = self.forward_propagation(X)
            self.back_propagation(X, y)
            self.update_parameters()
            
            if i % 100 == 0:
                print(f'Accuracy for iteration {i}: {self.get_accuracy(self.get_predictions(A2), y)}')

    # Methods for dev/test sets
    def predict(self, X):
        predictions = self.forward_propagation(X)
        return predictions
    
    def accuracy(self, predictions, y):
        return self.get_accuracy(self.get_predictions(predictions), y)
    