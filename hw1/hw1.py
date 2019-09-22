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
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


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
        x = 1 / (1 + np.exp(-x))
        self.state = x

        return self.state

    def derivative(self):

        # Maybe something we need later in here...

        return (1 - self.state) * self.state


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
        return 1 - self.state**2


class ReLU(Activation):

    """
    ReLU non-linearity
    """

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.state = x
        x[x < 0] = 0
        return x

    def derivative(self):
        x = self.state
        x[x < 0] = 0
        x[x != 0] = 1
        return x

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
        raise NotImplemented

    def derivative(self):
        raise NotImplemented


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
        
        logits = np.exp(self.logits)
        logits_sum = np.sum(logits, axis=1)
        self.sm = logits / logits_sum[:, None]
        loss = -np.sum(y * np.log(self.sm), axis=1)
        
        return loss

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

        self.x = x

        if eval:
            out = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma*out + self.beta
        else:
            self.mean = np.mean(x, axis=0)
            self.var = np.mean((x - self.mean)**2, axis=0)
            self.norm = (x - self.mean) / np.sqrt(self.var+self.eps)
            self.out = self.gamma*self.norm + self.beta

            self.running_mean = self.alpha*self.running_mean + (1-self.alpha)*self.mean
            self.running_var = self.alpha*self.running_var + (1-self.alpha)*self.var

            return self.out

        # self.mean = # ???
        # self.var = # ???
        # self.norm = # ???
        # self.out = # ???

        # update running batch statistics
        # self.running_mean = # ???
        # self.running_var = # ???

        # ...

        #raise NotImplemented

    def backward(self, delta):

        self.dgamma = np.sum(delta*self.norm, axis=0)
        self.dbeta = np.sum(delta, axis=0)

        dldxhat = delta * self.gamma
        dldsigma = - 1/2*np.sum(dldxhat*(self.x-self.mean)*((self.var+self.eps)**(-3/2)), axis=0)
        dldu = -np.sum(dldxhat*((self.var+self.eps)**(-1/2)), axis=0) - 2*dldsigma*np.mean(self.x-self.mean, axis=0)
        deriv = dldxhat*((self.var+self.eps)**(-1/2)) + dldsigma*2/self.x.shape[0]*(self.x-self.mean) + dldu/self.x.shape[0]

        return deriv / self.x.shape[0]


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
    return np.random.randn(d0, d1)


def zeros_bias_init(d):
    return np.zeros(d)


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
        
        self.inputs = [0] * (self.nlayers + 1)
        self.activation_inputs = [0] * self.nlayers
        self.shapes = [self.input_size] + hiddens + [self.output_size]
        
        
        self.dwdv = [0] * self.nlayers
        self.dbdv = [0] * self.nlayers
        for idx in range(self.nlayers):
            self.dwdv[idx] = np.zeros((self.shapes[idx], self.shapes[idx+1]))
        for idx in range(self.nlayers):
            self.dbdv[idx] = np.zeros(self.shapes[idx+1])
        
        
        # Don't change the name of the following class attributes,
        # the autograder will check against these attributes. But you will need to change
        # the values in order to initialize them correctly
        self.W = [0] * self.nlayers
        for idx in range(self.nlayers):
            self.W[idx] = weight_init_fn(self.shapes[idx], self.shapes[idx+1])
        self.dW = [0] * self.nlayers
        self.b = [0] * self.nlayers
        for idx in range(self.nlayers):
            self.b[idx] = bias_init_fn(self.shapes[idx+1])
        self.db = [0] * self.nlayers
        # HINT: self.foo = [ bar(???) for ?? in ? ]

        # if batch norm, add batch norm parameters
        self.bn_layers = []
        if self.bn:
            for i in range(1, self.num_bn_layers+1):
                self.bn_layers.append(BatchNorm(fan_in=self.shapes[i]))

        # Feel free to add any other attributes useful to your implementation (input, output, ...)

    def forward(self, x):
        self.inputs[0] = x
        
        for i in range(self.nlayers):
            
            z = np.dot(self.inputs[i], self.W[i]) + self.b[i]
            if i == 0:
                if i in range(self.num_bn_layers):
                    z = self.bn_layers[i].forward(z, not self.train_mode)
            self.activation_inputs[i] = z
            self.inputs[i+1] = self.activations[i](z)
            
        
        return self.inputs[-1]

    def zero_grads(self):
        for idx in range(self.nlayers):
            self.dW[idx] = np.zeros((self.shapes[idx], self.shapes[idx+1]))
        for idx in range(self.nlayers):
            self.db[idx] = np.zeros(self.shapes[idx+1])

    def step(self):
        for idx in range(self.nlayers):
            a = self.W[idx].copy()
            self.dwdv[idx] = self.momentum*self.dwdv[idx] - self.lr*self.dW[idx]
            self.W[idx] = a + self.dwdv[idx]
            b = self.b[idx].copy()
            self.dbdv[idx] = self.momentum*self.dbdv[idx] - self.lr*self.db[idx]
            self.b[idx] = b + self.dbdv[idx]
        
        for idx in range(len(self.bn_layers)):
            self.bn_layers[idx].gamma = self.bn_layers[idx].gamma - self.lr*self.bn_layers[idx].dgamma
            self.bn_layers[idx].beta = self.bn_layers[idx].beta - self.lr*self.bn_layers[idx].dbeta
        
        
    def backward(self, labels):
        self.zero_grads()
        loss = self.criterion(self.inputs[-1], labels)
        self.lol = loss
        deriv = self.criterion.derivative() / self.inputs[0].shape[0]
    
        for idx in range(self.nlayers-1, -1, -1):
            deriv = self.activations[idx].derivative() * deriv
            if self.bn and self.train_mode==True:
                if idx == 0:
                    deriv = self.bn_layers[0].backward(deriv * self.inputs[0].shape[0])
            self.dW[idx] = np.dot(self.inputs[idx].T, deriv)
            self.db[idx] = np.dot(deriv.T, np.ones((self.inputs[idx].shape[0], 1))).reshape(-1)
            deriv = np.dot(deriv, self.W[idx].T)

    def __call__(self, x):
        return self.forward(x)

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False



def get_training_stats(mlp, dset, nepochs, batch_size):

    train, val, test = dset
    trainx, trainy = train
    valx, valy = val
    testx, testy = test

    idxs = np.arange(len(trainx))

    training_losses = []
    training_errors = []
    validation_losses = []
    validation_errors = []
    test_losses = []
    test_errors = []
    training_losses_stats = []
    training_errors_stats = []
    validation_losses_stats = []
    validation_errors_stats = []

    np.random.seed(11785)
    #model = MLP(784, 10, [512, 128, 32], [ReLU(), ReLU(), ReLU(), Identity()], random_normal_weight_init, zeros_bias_init, SoftmaxCrossEntropy(), 0.01, momentum=0.9, num_bn_layers=0)
    model = mlp
    model.train()

    for e in range(nepochs):

        # Per epoch setup ...
        seed = np.random.randint(11785)
        np.random.seed(seed)
        np.random.shuffle(trainx)
        np.random.seed(seed)
        np.random.shuffle(trainy)
        
        seed = np.random.randint(11785)
        np.random.seed(seed)
        np.random.shuffle(valx)
        np.random.seed(seed)
        np.random.shuffle(valy)
        
        model.train()

        for b in range(0, len(trainx), batch_size):

            # Remove this line when you start implementing this
            # Train ...

            x_batch = trainx[b:b + batch_size]
            y_batch = trainy[b:b + batch_size]

            model.zero_grads()
            preds = model.forward(x_batch)
            model.backward(y_batch)
            loss = model.lol
            model.step()

            answers = np.argmax(preds, axis=1)
            labels = np.argmax(y_batch, axis=1)
            error = (answers[answers!=labels]).shape[0] / len(answers)

            training_losses_stats.append(loss)
            training_errors_stats.append(error)

        for b in range(0, len(valx), batch_size):

            # Remove this line when you start implementing this
            # Val ...

            model.eval()

            x_batch = valx[b:b + batch_size]
            y_batch = valy[b:b + batch_size]

            model.zero_grads()
            preds = model.forward(x_batch)
            loss = model.criterion(preds, y_batch)

            answers = np.argmax(preds, axis=1)
            labels = np.argmax(y_batch, axis=1)
            error = float(len(answers[answers!=labels])) / len(answers)
            
            validation_losses_stats.append(loss)
            validation_errors_stats.append(error)            


        # Accumulate data...
        training_losses.append(np.mean(training_losses_stats))
        training_errors.append(np.mean(training_errors_stats))
        
        validation_losses.append(np.mean(validation_losses_stats))
        validation_errors.append(np.mean(validation_errors_stats))
        

    # Cleanup ...
    model.eval()
    
    seed = np.random.randint(11785)
    np.random.seed(seed)
    np.random.shuffle(testx)
    np.random.seed(seed)
    np.random.shuffle(testy)    
    
    for b in range(0, len(testx), batch_size):

        # Remove this line when you start implementing this
        # Test ...

        x_batch = testx[b:b + batch_size]
        y_batch = testy[b:b + batch_size]

        model.zero_grads()
        preds = model.forward(x_batch)
        model.backward(y_batch)
        loss = model.criterion(model.inputs[-1], y_batch)

        answers = np.argmax(preds, axis=1)
        labels = np.argmax(y_batch, axis=1)
        error = len(answers[answers!=labels]) / len(answers)

        test_losses.append(loss)
        test_errors.append(error)

    # Return results ...

    # return (training_losses, training_errors, validation_losses, validation_errors)

    return training_losses, training_errors, validation_losses, validation_errors