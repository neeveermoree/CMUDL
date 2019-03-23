"""
Follow the instructions provided in the writeup to completely
implement the class specifications for a basic MLP, optimizer, .
You will be able to test each section individually by submitting
to autolab after implementing what is required for that section
-- do not worry if some methods required are not implemented yet.

Notes:

The __call__ method is a special reserved method in
python that defines the behaviour of an object when it is
used as a function. For example, take the Linear activation
function whose implementation has been provided.

# >>> activation = Identity()
# >>> activation(3)
# 3
# >>> activation.forward(3)
# 3
"""

# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np
import os


class Activation(object):

    """
    Interface for activation functions (non-linearities).

    In all implementations, the state attribute must contain the result, i.e. the output of forward (it will be tested).
    """

    # No additional work is needed for this class, as it acts like an abstract base class for the others

    def __init__(self):
        self.state = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        pass

    def derivative(self):
        pass


class Identity(Activation):

    """
    Identity function (already implemented).
    """

    # This class is a gimme as it is already implemented for you as an example

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        self.state = x
        return x

    def derivative(self):
        return 1.0


class Sigmoid(Activation):

    """
    Sigmoid non-linearity
    """

    # Remember do not change the function signatures as those are needed to stay the same for AL

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x):

        # Might we need to store something before returning?
        self.state = 1 / (1 + np.exp(-x))
        return self.state

    def derivative(self):

        # Maybe something we need later in here...
        return self.state * (1 - self.state)


class Tanh(Activation):

    """
    Tanh non-linearity
    """

    # This one's all you!

    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        self.state = np.tanh(x)
        return self.state

    def derivative(self):
        return 1 - np.power(self.state, 2) 


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = np.where(x < 0, 0, x)
        return self.state

    def derivative(self):
        return np.where(self.state > 0, 1, 0).astype(np.float64)

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

    """
    Interface for loss functions.
    """

    # Nothing needs done to this class, it's used by the following Criterion classes

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        pass

    def derivative(self):
        pass


class SoftmaxCrossEntropy(Criterion):

    """
    Softmax loss
    """

    def __init__(self):

        super(SoftmaxCrossEntropy, self).__init__()
        self.sm = None

    def forward(self, x, y):

        self.logits = x
        self.labels = y

        # ...
        self.sm = (np.exp(x).T / np.sum(np.exp(x), axis=1)).T
        self.loss = - np.sum(y * np.log(self.sm), axis=1)
        
        return self.loss

    def derivative(self):

        # self.sm might be useful here...
        return self.sm - self.labels

    
class BatchNorm(object):

    def __init__(self, fan_in, alpha=0.9):

        # You shouldn't need to edit anything in init

        self.alpha = alpha
        self.eps = 1e-8
        self.x = None
        self.norm = None
        self.out = None

        # The following attributes will be tested
        self.var = np.ones((1, fan_in))
        self.mean = np.zeros((1, fan_in))

        self.gamma = np.ones((1, fan_in))
        self.dgamma = np.zeros((1, fan_in))

        self.beta = np.zeros((1, fan_in))
        self.dbeta = np.zeros((1, fan_in))

        # inference parameters
        self.running_mean = np.zeros((1, fan_in))
        self.running_var = np.ones((1, fan_in))

    def __call__(self, x, eval=False):
        return self.forward(x, eval)

    def forward(self, x, eval=False):

        # if eval:
        #    # ???

        self.x = x

        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        # ...

        pass

    def backward(self, delta):

        pass


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.array([[np.random.normal() for _ in range(d1)] for __ in range(d0)])


def zeros_bias_init(d):
    return np.full(d, 0)


class MLP(object):

    """
    A simple multilayer perceptron
    """

    def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

        # Don't change this -->
        self.train_mode = True
        self.num_bn_layers = num_bn_layers
        self.bn = num_bn_layers > 0
        self.nlayers = len(hiddens) + 1
        self.input_size = input_size
        self.output_size = output_size
        self.activations = activations
        self.criterion = criterion
        self.lr = lr
        self.momentum = momentum
        # <---------------------
        self.weight_fn = weight_init_fn
        self.bias_fn = bias_init_fn
        self.output = []
        self.layers = [input_size] + hiddens + [output_size]
        self.bias = []
        
        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        # self.W = None
        self.W = []
        self.b = []
        self.dW = []
        self.db = []
        for i in range(self.nlayers):
            self.W.append(self.weight_fn(self.layers[i], self.layers[i+1]))
            self.b.append(self.bias_fn(self.layers[i+1]))
            self.dW.append(np.zeros((self.layers[i], self.layers[i+1])))
            self.db.append(np.zeros(self.layers[i+1]))
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        if self.bn:
            self.bn_layers = None

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        self.batch_size = x.shape[0]
        self.input = x
        for i in range(self.nlayers):
            self.bias.append(np.ones(self.batch_size))
        
        if len(self.W) < self.nlayers:
            for i in range(self.nlayers - len(self.W)):
                self.W.append(self.weight_fn(self.layers[i], self.layers[i+1]))
                self.b.append(self.bias_fn(self.layers[i+1]))
                
        for i in range(self.nlayers):
            if i == 0:
                output = np.dot(self.W[i].T, x.T) + self.b[i].T
            else:
                output = np.dot(self.W[i].T, self.output[i-1]) + self.b[i].T
            self.output.append(self.activations[i].forward(output))
            
        return self.output[-1].T

    def zero_grads(self):
        for i in range(self.nlayers):
            self.dW[i] = np.zeros((self.layers[i], self.layers[i+1]))
            self.db[i] = np.zeros(self.layers[i+1])

    def step(self):
        for i in range(self.nlayers - 1, -1, -1):
            self.W[i] -= self.dW[i] * self.lr
            self.b[i] -= self.db[i] * self.lr 
        
    def backward(self, labels):
        
        loss = self.criterion.forward(self.output[-1].T, labels)
        self.deriv = self.criterion.derivative()
        
        if self.nlayers == 1:
            activ_deriv = self.activations[0].derivative()
            self.dW[0] = np.dot(self.input.T, activ_deriv * self.deriv) / self.batch_size
            self.db[0] = np.dot(self.bias[0], activ_deriv * self.deriv) / self.batch_size
        
        if self.nlayers == 2:
            for i in range(self.nlayers - 1, -1, -1):
                activ_deriv = self.activations[i].derivative()
                if i == self.nlayers - 1:
                    self.deriv = activ_deriv * self.deriv
                    self.dW[i] = np.dot(self.output[-2], self.deriv) / self.batch_size
                    self.db[i] = np.dot(self.bias[i], self.deriv) / self.batch_size
                elif i == 0:
                    self.dW[i] = (np.dot(np.dot(self.deriv, self.W[1].T).T * activ_deriv, self.input) / self.batch_size).T       
                    self.db[i] = np.dot(np.dot(self.deriv, self.W[1].T).T * activ_deriv, self.bias[0]) / self.batch_size
        
        if self.nlayers > 2:
            out = -2
            for i in range(self.nlayers - 1, -1, -1):
                activ_deriv = self.activations[i].derivative()
                if i == self.nlayers - 1:
                    self.deriv = activ_deriv * self.deriv
                    self.dW[i] = np.dot(self.output[out], self.deriv) / self.batch_size
                    self.db[i] = np.dot(self.bias[i], self.deriv) / self.batch_size
                    self.deriv = np.dot(self.W[i], self.deriv.T)
                elif i == 0:
                    #l = np.dot(self.W[i+1].T, self.deriv).T
                    self.dW[i] = (np.dot((self.deriv * activ_deriv), self.input) / self.batch_size).T       
                    self.db[i] = np.dot(self.deriv * activ_deriv, self.bias[0]) / self.batch_size
                else:
                    self.deriv = activ_deriv * self.deriv
                    self.dW[i] = np.dot(self.deriv, self.output[out].T).T / self.batch_size
                    self.db[i] = np.dot(self.deriv, self.bias[i]) / self.batch_size
                    self.deriv = np.dot(self.W[i], self.deriv)
                out -= 1
                
    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

if __name__ == "__main__":

    import pickle

    data = pickle.load(open('../local_autograder/data.pkl', 'rb'))

    train1 = data[16][0]
    train2 = data[16][1]
    ans1 = data[16][2:5]
    ans2 = data[16][5:]

    def weight_init(x, y):
        return np.random.randn(x, y)


    def bias_init(x):
        return np.zeros((1, x))


    mlp = MLP(784, 10, [64, 32], [Sigmoid(), Sigmoid(), Identity()],
                      weight_init, bias_init, SoftmaxCrossEntropy(), 0.008,
                      momentum=0.0, num_bn_layers=0)
    mlp.forward(train1)
    mlp.backward(train2)

