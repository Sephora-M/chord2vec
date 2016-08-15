import numpy as np
from numpy import linalg as LA

def sigmoid_cross_entropy_cost(outputs, targets, derivative=False, epsilon=1e-8):
    """
    the sigmoid cross entropy cost
    """
    # preventing overflow
    outputs = np.clip(outputs, epsilon, 1 - epsilon)
    divisor = np.maximum(outputs * (1 - outputs), epsilon)

    if derivative:
        return (outputs - targets) / divisor
    else:
        return np.mean(-np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=1))

expit = lambda x: 1.0 / (1 + np.exp(-x))

def sigmoid_function(signal, derivative=False):
    """
    Sigmoid function
    Args:
        signal:
        derivative:

    Returns:

    """

    # preventing overflow.
    signal = np.clip(signal, -500, 500)

    # Calculate activation signal
    signal = expit(signal)

    if derivative:
        # Return the partial derivation of the activation function
        return np.multiply(signal, 1 - signal)
    else:
        # Return the activation signal
        return signal

def linear_function( signal, derivative=False ):
    if derivative:
        # Return the partial derivation of the activation function
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal

def normalize_function(signal, derivative=False):
    if derivative:
        # Return the partial derivation of the activation function
        return np.ones( signal.shape )
    else:
        # Return the activation signal
        return signal
