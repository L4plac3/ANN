import numpy as np
import time
from tqdm import tqdm

class ANN():
    '''
    Class for a general Artificial Neural Network 
    used for regression purposes.
    '''

    def __init__(self, layers):
        '''
        Class constructor.

        Parameters:
            - layers: list containing the number of units for each layer
        
        Example
        net.ANN([8,10,10,1]) represents a network having:
                - 8 input units
                - 2 hidden layers having 10 hidden units each
                - 1 output unit
        '''
        self.L = len(layers) - 1
        self.layers = layers
        self.initialization()
        
    def initialization(self):
        '''
        Initializes the weights of the network to a random value 
        according to a Gaussian distribution centered in 0.
        '''
        self.W = np.array([0] * (self.L + 1), dtype=object)
        self.b = np.array([0] * (self.L + 1), dtype=object)
        for l in range(1, self.L + 1):
            self.W[l] = np.random.randn(self.layers[l], self.layers[l - 1])
            self.b[l] = np.random.randn(self.layers[l],)

    def normalization(self, x, target, target_norm=True):
        '''
        Normalization of the input in the range [0,1].
        '''
        self.x = x / x.max(axis=0)
        if target_norm:
            self.target = target / target.max(axis=0)
        else: self.target = target
    
    def split_train_test(self, x, target, train_set=0.85, shuffle=True, target_norm=True):
        '''
        Splits the dataset into training and testing sets according to the % of data that we 
        want to be in the training set (train_set=0.85).
        If shuffle is True the dataset is first randomly shuffled and then the splitting is applied.
        If target_norm is True, also the target is normalized.
        '''
        self.normalization(x, target, target_norm)
        tmp = np.concatenate((self.x, self.target.T), axis=1)
        if shuffle:
            np.random.shuffle(tmp)
        n_train = int(np.floor(x.shape[0]*train_set))
        train = tmp[:n_train]
        test = tmp[n_train:]
        self.x_train = train[:,:-1]
        self.target_train = train[:,-1]
        self.x_test = test[:,:-1]
        self.target_test = test[:,-1]

    def sigmoid(self, a):
        '''
        Sigmoid activation function.
        '''
        return 1.0 / (1.0 + np.exp(-a))

    def sigmoid_derivative(self, a):
        '''
        Derivative of the sigmoid wrt its argument.
        '''
        return (1.0 - self.sigmoid(a)) * self.sigmoid(a)

    def relu(self, a):
        '''
        Rectifier Linear Unit activation function.
        '''
        return np.maximum(0, a)

    def relu_derivative(self, a):
        '''
        Derivative of the ReLU wrt its argument.
        '''
        a[a<=0] = 0
        a[a>0] = 1
        return a

    def loss(self, target, output):
        '''
        Loss function: MSE (Mean Squared Error)
        '''
        diff = target - output
        return 0.5 * np.dot(diff, diff.T)

    def forward(self, x):
        '''
        Forward step of the backpropagation algorithm.
        It forwards the input toward the output unit passing through the hidden layers.
        It computes the output of each unit in the network wrt the current input x.
        The ReLU is applied to the hidden units, while the sigmoid to the output unit.
        '''
        self.a = np.array([0] * (self.L + 1), dtype=object)
        self.o = np.array([0] * (self.L + 1), dtype=object)
        self.o[0] = np.copy(x)
        for l in range(1, self.L + 1):
            self.a[l] = np.dot(self.W[l], self.o[l - 1]) + self.b[l]
            if l == self.L:
                self.o[l] = self.sigmoid(self.a[l])
            else:
                self.o[l] = self.relu(self.a[l])

    def backward(self, target):
        '''
        Backward step of the backpropagation algorithm.
        It computes the delta errors of each unit in the network, wrt to the current input x,
        that has been forward propagated via the self.forward step.
        '''
        self.deltas = np.array([0] * (self.L + 1), dtype=object)
        for l in reversed(range(1, self.L + 1)):
            if l == self.L:
                self.deltas[l] = self.sigmoid_derivative(self.a[l]) * (self.o[l] - target)
            else:
                self.deltas[l] = self.relu_derivative(self.a[l]) * np.dot(self.deltas[l + 1].T, self.W[l + 1])

    def loss_gradient(self):
        '''
        Computes the derivative of the empirical error wrt to all the weights in the network and
        stores them in a matrix dE.
        '''
        self.dE = np.array([0] * (self.L + 1), dtype=object)
        for l in range(1, self.L + 1):
            self.dE[l] = np.outer(self.deltas[l], self.o[l-1])

    def weight_update(self, learning_rate, dE):
        '''
        Weight update rule according to gradient descent method.
        w = w - learning_rate*dE
        '''
        self.W = self.W - learning_rate * dE

    def backpropagation(self, x, target):
        '''
        Backpropagation algorithm.
        '''
        self.forward(x)
        self.backward(target)
        self.loss_gradient()
        
    def train(self, epochs=10000, eta=0.1):
        '''
        Training of the network.

        Parameters:
            - epochs: number of epochs
            - eta: learning_rate
        '''
        for _ in tqdm(range(epochs)):
            for i in range(len(self.x_train)):
                self.backpropagation(self.x_train[i], self.target_train[i])
                self.weight_update(eta, self.dE)
        
    def test(self):
        '''
        Test of the network.
        '''
        error = 0
        for i in range(len(self.x_test)):
            self.forward(self.x_test[i])
            error += self.loss(self.o[self.L], self.target_test[i])
        E = error/len(self.x_test) 
        print('Accuracy: {}%'.format(round((1-E)*100,2)))