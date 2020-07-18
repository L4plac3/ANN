import numpy as np

class ANN():

    def __init__(self, layers):
        self.L = len(layers) - 1
        self.layers = layers
        self.initialization()
        
    def initialization(self):
        self.W = np.array([0] * (self.L + 1), dtype=object)
        # self.b = np.array([0] * (self.L + 1), dtype=object)
        for l in range(1, self.L + 1):
            self.W[l] = np.random.rand(self.layers[l], self.layers[l - 1])
            # self.b[l] = np.random.randn(self.layers[l],)

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

    def loss(self, target, output):
        diff = target - output
        return 0.5 * np.dot(diff, diff.T)

    def forward(self, x):
        self.a = np.array([0] * (self.L + 1), dtype=object)
        self.o = np.array([0] * (self.L + 1), dtype=object)
        self.o[0] = np.copy(x)
        for l in range(1, self.L + 1):
            # self.a[l] = np.dot(self.W[l], self.o[l - 1]) + self.b[l]
            self.a[l] = np.dot(self.W[l], self.o[l - 1])
            if l == self.L:
                self.o[l] = self.sigmoid(self.a[l])
            else:
                self.o[l] = self.relu(self.a[l])

    def backward(self, target):
        self.deltas = np.array([0] * (self.L + 1), dtype=object)
        for l in reversed(range(1, self.L + 1)):
            if l == self.L:
                self.deltas[l] = self.sigmoid_derivative(self.a[l]) * (self.o[l] - target)
            else:
                self.deltas[l] = self.relu_derivative(self.a[l]) * np.dot(self.deltas[l + 1].T, self.W[l + 1])

    def loss_gradient(self):
        self.dE = np.array([0] * (self.L + 1), dtype=object)
        for l in range(1, self.L + 1):
            self.dE[l] = np.outer(self.deltas[l], self.o[l-1])

    def weight_update(self, learning_rate, dE):
        self.W = self.W - learning_rate * dE

    def normalization(self, x, target):
        norm_x = np.linalg.norm(x)
        norm_target = np.linalg.norm(target)
        self.x = x/norm_x
        self.target = target/norm_target

    def backpropagation(self, x, target):
        if len(x.shape) == 1:
            n = 1
            self.forward(x)
            self.backward(target)
            self.loss_gradient()
        else:
            n = x.shape[1]
            for i in range(len(x)):
                self.forward(x[i])
                self.backward(target[i])
                self.loss_gradient()
                if i == 0:
                    dE = self.dE
                else:
                    dE += self.dE
            return dE
        return self.dE
    
    def split_train_test(self, x, target, train_set=0.85, shuffle=True):
        self.normalization(x, target)
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
        
    def train(self, epochs=100000, learning_rate=0.1):
        for epoch in range(epochs):
            if epoch % 10000 == 0:
                print('{}%'.format(epoch/1000))
            for i in range(len(self.x_train)):
                dE = self.backpropagation(self.x_train[i], self.target_train[i])
                self.weight_update(learning_rate, dE)
        
    def test(self, x, target):
        self.forward(x)
        error = self.loss(self.o[self.L], target)
        return error
                
    def print_all(self):
        attrs = vars(self)
        print('\n\n'.join("%s: \n%s" % item for item in attrs.items()))