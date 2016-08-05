import numpy as np

import math
NUM_NOTES = 88
class LinearModel1(object):
    """Linear Chord2vec model
    """
    def __init__(self, inputs, hidden_units):
        """Create the model

        """

    def step(self, inputs, targets, forward_only = False):
        """

        Args:
            inputs:
            targets:
            forward_only:

        Returns:

        """

def sigmoid_cross_entropy_loss():

    def test_check_grad(self):

        n = 4
        A = np.random.randn(n * n).reshape(n, n)
        b = np.random.randn(n).reshape((-1, 1))
        f = lambda x: np.dot(x.T, np.dot(A, x)) + np.dot(b.T, x)
        df = lambda x: np.dot(A + A.T, x) + b
        hf = lambda x: A + A.T
        n_runs = 100
        max_fails = int(n_runs * 0.05)
        n_fails = 0
        for _ in range(n_runs):
            check_grad(np.random.randn(n).reshape(-1, 1), f, df)
            if not check_grad(np.random.randn(n), f, hf, numdiff=numerical_hess, atol=1e-2, do_raise=False):
                n_fails += 1
        if n_fails > max_fails:
            raise Exception('%i fails' % n_fails)

def numerical_grad(x, f, eps=None):
    if eps is None:
        eps = 1e-6
    fx = f(x)
    flatten = hasattr(fx, 'flatten')
    nx = np.prod(x.shape)
    nf = np.prod(fx.shape)
    df = np.zeros((nx, nf), dtype=float)
    x = x * 1.0
    for i in range(len(x)):
        x[i] += eps
        fxi = f(x)
        x[i] -= eps
        if flatten:
            df[i, :] = (fxi - fx).flatten() / eps
        else:
            df[i, :] = (fxi - fx) / eps
    return df

def numerical_hess(x, f, eps=None):
    if eps is None:
        eps = 1e-6
    g = lambda x: numerical_grad(x, f, eps=eps)
    return numerical_grad(x, g, eps=eps)

def check_grad(x, f, df, eps=1e-6, atol=1e-5, warn=False, numdiff=numerical_grad, do_raise=True):
    df_anal = df(x)
    df_num = numdiff(x, f, eps=eps)
    df_num = df_num.reshape(df_anal.shape)
    if not np.allclose(df_anal, df_num, atol=atol):
        s = '\nanal, num, ratio, diff:\n' + '\n'.join(
            map(str, ([x.flatten() for x in [df_anal, df_num, df_anal / df_num, df_anal - df_num]])))
        if warn:
            print(s)
        else:
            if do_raise:
                raise Exception(s)
            else:
                return False
    return True
