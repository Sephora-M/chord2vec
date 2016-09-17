import numpy as np
from numpy import linalg as LA

def binary_cross_entropy_cost(outputs, targets, derivative=False, epsilon=1e-8):
    """
    Binary cross entropy cost
    """
    # preventing overflow
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    divisor = np.maximum(outputs * (1 - outputs), epsilon)
    if derivative:
        #derivative wrt output
        return (outputs - targets) / divisor
    else:
        return np.mean(-np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=1))

expit = lambda x: 1.0 / (1 + np.exp(-x))

def sigmoid_function(signal, derivative=False):
    """
    Sigmoid function

    """
    # preventing overflow.
    signal = np.clip(signal, -500, 500)

    # element wise sigmoid
    signal = expit(signal)

    if derivative:
        # derivative wrt signal
        return np.multiply(signal, 1 - signal)
    else:
        return signal

def linear_function( signal, derivative=False ):
    if derivative:
        # derivative wrt signal
        return np.ones( signal.shape, float )
    else:
        return signal


def zero_padding(weights_layer):
    """
    Returns a triangular matrix with all ones in the upper triangle
    """
    num_units = weights_layer.shape[1]
    padding = np.zeros((num_units,num_units), float)
    for i in range(num_units):
        for j in range(num_units):
            if i < j:
                padding[i][j] = 1.0
    return padding

def normalize_function(signal, derivative=False):
    """
    Normalize signal by dividing the signal with the L1 norm

    """
    if derivative:
        return np.ones( signal.shape )
    else:
        return np.array([s/ (LA.norm(s, ord=1)) if (LA.norm(s, ord=1))>0 else s for s in signal])

def normalize(signal, derivative=False):
    """
    Identical as normalize_function, but returns lists

    """
    if derivative:
        # Return the partial derivation of the activation function
        ones = [[]]*signal.shape[0]
        ones[0].append([1]*signal.shape[1])
        return ones
    else:
        # Return the normalized  signal

        return ([s/ (LA.norm(s, ord=1)) if (LA.norm(s, ord=1))>0 else s for s in signal])
