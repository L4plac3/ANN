import numpy as np

class ANN():

    def __init__(self, layers):
        self.L = len(layers) - 1
        self.layers = layers
        self.initialization()
        
    def initialization(self):
        self.W = [0] * (self.L + 1)
        self.b = [0] * (self.L + 1)
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.b[l] = np.random.randn(self.layers[l],)

    def sigmoid(self, a):
        return 1.0 / (1.0 + np.exp(-a))

    def sigmoid_derivative(self, a):
        return (1.0 - self.sigmoid(a)) * self.sigmoid(a)

    def relu(self, a):
        return np.maximum(0, a)

    def relu_derivative(self, a):
        a[a<=0] = 0
        a[a>0] = 1
        return a

    def error(self, target, output):
        diff = target - output
        return 0.5 * np.dot(diff, diff.T)

    def forward(self, x):
        self.a = [0] * (self.L + 1)
        self.o = [0] * (self.L + 1)
        self.o[0] = np.copy(x)
        for l in range(1, self.L + 1):
            self.a[l] = np.dot(self.W[l], self.o[l - 1]) + self.b[l]
            if l == self.L:
                self.o[l] = self.sigmoid(self.a[l])
            else:
                self.o[l] = self.relu(self.a[l])
        return self.o[self.L]

    def backward(self, target):
        dE = [0] * (self.L + 1)
        self.deltas = [0] * (self.L + 1)
        for l in reversed(range(1, self.L + 1)):
            if l == self.L:
                self.deltas[l] = self.sigmoid_derivative(self.a[l]) * (self.o[l] - target)
                dE[l] = np.outer(self.deltas[l], self.o[l-1])
            else:
                self.deltas[l] = self.relu_derivative(self.a[l]) * np.dot(self.deltas[l + 1].T, self.W[l + 1])
                dE[l] = np.outer(self.deltas[l], self.o[l-1])
        return dE

    def backpropagation(self, x, target):
        error = 0
        if len(x.shape) == 1:
            n = 1
            output = self.forward(x)
            error += self.error(output, target)
            dE = self.backward(target)
        else:
            n = x.shape[1]
            for i in range(len(x)):
                output = self.forward(x[i])
                error += self.error(output, target[i])
                if i == 0:
                    dE = self.backward(target[i])
                else:
                    dE += self.backward(target[i])
        return error/n, dE

    # def train(self, x, target, batch_size=1):




        