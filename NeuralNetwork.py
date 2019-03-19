import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    return x * (1 - x)

class NeuralNetwork():

    def __init__(self, neuronsPerLayer):
        self.weights = list()
        self.bias = list()
        self.activationFunction = list()
        self.momentum = list()
        self.learningRate = 0.1
        self.momentumRate = 0.1

        for i in range(1, len(neuronsPerLayer)):
            weights = np.random.normal(0.0, pow(neuronsPerLayer[i - 1], -0.5), (neuronsPerLayer[i], neuronsPerLayer[i - 1]))
            bias = np.random.normal(0.0, 0.2, (neuronsPerLayer[i], 1))
            momentum = np.zeros((neuronsPerLayer[i], 1))
            
            self.bias.append(bias)
            self.momentum.append(momentum)
            self.activationFunction.append((sigmoid, dsigmoid))
            self.weights.append(weights)
        

    def predict(self, xs, history = list()):
        currentOuput = np.reshape(xs, (len(xs), 1))
        history.append(currentOuput)

        for i in range(0, len(self.weights)):
            currentOuput = np.dot(self.weights[i], currentOuput)
            currentOuput += self.bias[i]
            currentOuput = self.activationFunction[i][0](currentOuput)
            history.append(currentOuput)

        return np.reshape(currentOuput, (len(currentOuput)))

    def fit(self, xs, ys):
        history = list()
        self.predict(xs, history)
        desired = np.reshape(ys, (len(ys), 1))
        error = desired - history[len(history) - 1]

        newError = None
        for i in range(len(self.weights) - 1, -1, -1):
            if i >= 0:
                newError = np.dot(self.weights[i].T, error)
            
            gradient = error * self.activationFunction[i][1](history[i + 1])
            self.bias[i] += gradient * self.learningRate
            deltaWeight = self.learningRate * np.dot(gradient, history[i].T)
            deltaWeight += self.momentum[i]
            self.momentum[i] = deltaWeight * self.momentumRate
            self.weights[i] += deltaWeight
            error = newError
