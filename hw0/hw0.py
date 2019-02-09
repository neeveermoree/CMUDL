import numpy as np
import os
import time
import torch    
    


def sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array.
    y is a 1-dimensional int numpy array.
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    result = 0
    for i in range(len(x)):
        for j in range(len(y)):
            result += x[i] * y[j]
    return result




def vectorize_sumproducts(x, y):
    """
    x is a 1-dimensional int numpy array. Shape of x is (N, ).
    y is a 1-dimensional int numpy array. Shape of y is (N, ).
    Return the sum of x[i] * y[j] for all pairs of indices i, j.

    >>> vectorize_sumproducts(np.arange(3000), np.arange(3000))
    20236502250000

    """
    # Write the vecotrized version here
    arr = np.array([])
    for i in np.nditer(x):
        arr = np.append(arr, np.sum(y * i))
    return np.sum(arr.flat)


def Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
    return result

def vectorize_Relu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if < 0 else x[i][j] for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    np.place(x, x < 0, 0)
    return x


def ReluPrime(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    result = np.copy(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i][j] < 0:
                result[i][j] = 0
            else:
                result[i][j] = 1
    return result


def vectorize_PrimeRelu(x):
    """
    x is a 2-dimensional int numpy array.
    Return x[i][j] = 0 if x[i][j] < 0 else 1 for all pairs of indices i, j.

    """
    # Write the vecotrized version here
    return (x >= 0).astype(int)  


def slice_fixed_point(x, l, start_point):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should have.
    start_point is an integer representing the point at which the final utterance should start in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    arr = np.array([])
    for i in range(x.shape[0]):
        arr = np.append(arr, x[i][start_point:start_point+l])
    return arr.reshape(x.shape[0], l, -1)


def slice_last_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    arr = np.array([])
    for i in range(x.shape[0]):
        arr = np.append(arr, x[i][-l:])
    return arr.reshape(x.shape[0], l, -1)


def slice_random_point(x, l):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    arr = np.array([])
    for i in range(x.shape[0]):
        a = np.array([])
        r = np.random.randint(len(x[i]) - l + 1)
        for j in range(r, r + l):
            a = np.append(a, x[i][j])
        arr = np.append(arr, a)
    return arr.reshape(x.shape[0], l, -1)


def pad_pattern_end(x):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    n = x.shape[0]
    max_length = 0
    arr = np.array([])
    for i in np.arange(n):
        obj = x[i].shape[0]
        if max_length < obj:
            max_length = obj
    for i in np.arange(n):
        razn = max_length - x[i].shape[0]
        x[i] = np.pad(x[i], ((0, razn), (0, 0)), mode='symmetric')
    return np.stack(x)
    


def pad_constant_central(x, c_):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    max_length = 0
    n = x.shape[0]
    for i in np.arange(n):
        if max_length < x[i].shape[0]:
            max_length = x[i].shape[0]
    for i in np.arange(n):
        razn = max_length - x[i].shape[0]
        if razn % 2 == 0:
            l = razn / 2
            r = razn / 2
        else:
            l = razn / 2 - 0.5
            r = razn / 2 + 0.5
        x[i] = np.pad(x[i], ((int(l), int(r)), (0, 0)), mode='constant', constant_values=c_)
    return np.stack(x)



def numpy2tensor(x):
    """
    x is an numpy nd-array. 

    Return a pytorch Tensor of the same shape containing the same data.
    """
    return torch.from_numpy(x)

def tensor2numpy(x):
    """
    x is a pytorch Tensor. 

    Return a numpy nd-array of the same shape containing the same data.
    """
    return x.numpy()

def tensor_sumproducts(x,y):
    """
    x is an n-dimensional pytorch Tensor.
    y is an n-dimensional pytorch Tensor.

    Return the sum of the element-wise product of the two tensors.
    """
    return torch.sum(x * y)

def tensor_ReLU(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the ReLU function: 
    RELU(i) = 0 if i < 0 else i

    Return a pytorch Tensor of the same shape as x containing RELU(x)
    """
    x[x < 0] = 0
    return x        

def tensor_ReLU_prime(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the RELU_PRIME function: 
    RELU_PRIME(i) = 0 if i < 0 else 1

    Return a pytorch Tensor of the same shape as x containing RELU_PRIME(x)
    """
    x[x >= 0] = 1
    x[x < 0] = 0
    return x
