import numpy as np
import os
import time
import torch    
    

def outer(x, y):
    """
    Compute the outer product of two vectors.

    Parameters: 
    x (numpy.ndarray): 1-dimensional numpy array.
    y (numpy.ndarray): 1-dimensional numpy array.

    Returns: 
    numpy.ndarray: 2-dimensional numpy array.
    """

    return np.matmul([x.reshape(-1, 1)], [y.reshape(1, -1)])


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
    ones = np.ones(len(x))
    ones_column_vector = ones.reshape(1, -1)
    out = outer(x, y).reshape(len(x), -1)
    first_product = np.matmul(ones_column_vector, out)
    second_product = np.matmul(first_product, ones.reshape(-1, 1))
    return second_product[0].astype(np.int64)[0]


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
    x[x<0] = 0
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
    x[~(x<0)] = 1
    x[x<0] = 0
    return x

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


def slice_last_point(x, m):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    # Pre-define output function dimension specification
    dim1 = spectrograms.shape[0]    # n
    dim2 = m                       # m
    dim3 = spectrograms[0].shape[1] # k

    result = np.zeros((dim1,dim2,dim3))

    #### Start of your code ####
    lengths = [len(spectro) for spectro in spectrograms]
    result = np.array([arr[lengths[i]-m:] for i, arr in enumerate(x)])
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result


def slice_random_point(x, d):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.
    l is an integer representing the length of the utterances that the final array should be in.
    Return a 3-dimensional int numpy array of shape (n, l, -1)

    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)
        assert(utter.shape[0] >= d)

    offset = [np.random.randint(utter.shape[0]-d+1)
              if utter.shape[0]-d > 0 else 0
              for utter in spectrograms]

    # Pre-define output function dimension specification
    dim1 = spectrograms.shape[0]    # n
    dim2 = d                       # d
    dim3 = spectrograms[0].shape[1] # k

    result = np.zeros((dim1,dim2,dim3))

    #### Start of your code ####
    lengths = [len(spectro) for spectro in spectrograms]
    random_points = [np.random.randint(i-d+1) for i in lengths]
    result = np.array([arr[offset[i]:offset[i]+d] for i, arr in enumerate(x)])
    #print(offset, random_points)
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result


def pad_pattern_end(x):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    # Pre-define output function dimension specification
    dim1 = spectrograms.shape[0]    # n
    dim2 = max([utter.shape[0] for utter in spectrograms]) # m
    dim3 = spectrograms[0].shape[1] # k

    result = np.zeros((dim1, dim2, dim3))

    #### Start of your code ####
    lengths = np.array([len(spectro) for spectro in spectrograms])
    max_length = np.array([max(lengths) for i in range(dim1)])
    paddings = max_length - lengths
    
    pad_above = 0
    pad_left = 0
    pad_right = 0

#     n_add = [((pad_above, pad_below1), (pad_left, pad_right)), 
#              ((pad_above, pad_below2), (pad_left, pad_right)), 
#              ((pad_above, pad_below3), (pad_left, pad_right))]

    n_add = [((pad_above, pad_below), (pad_left, pad_right)) for pad_below in paddings]
    result = [np.pad(x[i], pad_width=n_add[i], mode='symmetric') for i in range(dim1)]
    result = np.array(result)
    
    
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result
    


def pad_constant_central(x, cval):
    """
    x is a 3-dimensional int numpy array, (n, ). First dimension represent the number of instances in the array.
    Second dimension is variable, depending on the length of a given instance. Third dimension is fixed
    to the number of features extracted per utterance in an instance.

    Return a 3-dimensional int numpy array.

    """
    spectrograms = x
    
    # Input function dimension specification
    assert(spectrograms.ndim == 1)
    for utter in spectrograms:
        assert(utter.ndim == 2)

    dim1 = spectrograms.shape[0]    # n
    dim2 = max([utter.shape[0] for utter in spectrograms]) # m
    dim3 = spectrograms[0].shape[1] # k

    result = np.ones((dim1,dim2,dim3))

    #### Start of your code ####

    lengths = np.array([len(spectro) for spectro in spectrograms])
    max_length = np.array([max(lengths) for i in range(dim1)])
    paddings = max_length - lengths
    
    pad_right = pad_left = 0
    pad_top = paddings // 2
    pad_below = paddings % 2 + pad_top
    
#     print(pad_top.shape)
#     return NotImplemented
    n_add = [((pad_top[i], pad_below[i]), (pad_left, pad_right)) for i in range(dim1)]
    result = [np.pad(x[i], pad_width=n_add[i], mode='constant', constant_values=cval) for i in range(dim1)]
    result = np.array(result)
    
    ####  End of your code  ####

    # Assert output function dimension specification
    assert(result.shape[0] == dim1)
    assert(result.shape[1] == dim2)
    assert(result.shape[2] == dim3)
    
    return result



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
    x[x<0] = 0
    return x        

def tensor_ReLU_prime(x):
    """
    x is a pytorch Tensor. 
    For every element i in x, apply the RELU_PRIME function: 
    RELU_PRIME(i) = 0 if i < 0 else 1

    Return a pytorch Tensor of the same shape as x containing RELU_PRIME(x)
    """
    x[~(x<0)] = 1
    x[x<0] = 0
    return x
