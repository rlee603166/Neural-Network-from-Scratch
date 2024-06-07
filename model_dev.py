import numpy as np
import pandas as pd
np.random.seed(42)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

X_train = np.array(train.iloc[:,1:]).T
y_train = train['label'].to_numpy().reshape(-1,1)

def label_encoder(y):
    zeros = np.zeros((y.shape[0], np.max(y)+1))
    for i, num in enumerate(y):
        zeros[i][np.squeeze(num)] = 1
    return zeros

y_train = label_encoder(y_train)
print(y_train[:-5])

def initialize_parameters(n_x, n_h=10, n_y=10):
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

def relu(Z):
    return np.maximum(0,Z)

def sigmoid(Z):
    sigmoid = 1 / (1 + np.exp(-Z))
    return sigmoid

def softmax(Z):
    softmax = np.exp(x)/sum(np.exp(x))
    return np.sum(softmax, axis= 1, keepdims=True)

def forward_propagation(X, params):
    W1, b1, W2, b2 = params.values()
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
      }
        self.parameters['W1'] = W1 - learning_rate * self.gradients['dW1']
    return A2, cache

def compute_cost(A2, y):
    m = y.shape[0]
    
    cross_entropy = np.multiply(np.log(A2).T, y)
    cost = - np.sum(cross_entropy, axis=0)
    return cost

def back_propagation(params, cache, X, y):
    W1, b1, W2, b2 = params.values()
    Z1, A1, Z2, A2 = cache.values()

    m = X.shape[1]
    
    dZ2 = A2 - y.T
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grad_params = {
        'dZ2': dZ2,
        'dW2': dW2,
        'db2': db2,
        'dZ1': dZ1,
        'dW1': dW1,
        'db1': db1

    }

    return grad_params

def update_parameters(params, gradients, learning_rate):
    params['W1'] -= learning_rate * gradients['dW1']
    params['b1'] -= learning_rate * gradients['db1']
    params['W2'] -= learning_rate * gradients['dW2']
    params['b2'] -= learning_rate * gradients['db2']

def model(X, y, learning_rate = 0.01, num_iterations = 3000):
    parameters = initialize_parameters(X.shape[0])
    costs = []
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

parameters, costs = model(X_train, y_train)