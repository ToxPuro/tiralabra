import numpy as np
from helpers import *


def affine_forward(x, w, b):
    """Computes the forward pass for an affine layer.
    -x: input data
    -w: weights of the layer
    -b: biases of the layer

    -out: output of the layer
    -cache: cache for calculating the gradients in backpropagation
    """
    ## manipulate into correct shape
    x_reshape = reshape(x)
    out = np.dot(x_reshape, w) + b

    ## affine means w*x+b
    out = np.dot(x_reshape, w) + b

    ##cache values
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine layer.
    
    - dout: Upstream derivatives
    - cache: cache from affine_forward

    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    ## values from affine_forward cache
    x, w, b = cache

    ##manipulate into correct shape
    x_reshape = reshape(x)

    ## w*x+b derivative respect to x is x, how to make sure is in correct shape
    dx = dout.dot(w.T).reshape(x.shape)

    ##w*x+b derivative respect to w is x
    dw = x_reshape.T.dot(dout)

    ##w*x+b derivative respect to b is 1, sum because there are N samples in one batch
    db = np.sum(dout, axis=0)
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of ReLUs.

    - x: input
    - out: output meaning Relu(x)
    - cache: cache for calculating the gradients in backprop
    """

    out = np.maximum(0,x)
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of ReLUs.

    - dout: Upstream derivatives
    - cache: cache from relu_forward
    - dx: Gradient with respect to x
    
    """

    ## values from relu_forward
    dx, x = None, cache

    ## derivative of max(0,x) is 1 for when x>0 and 0 otherwise
    dx = np.ones(x.shape)
    dx[x<=0] = 0

    dx = dx*dout
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax.

    Inputs:
    - x: Input data
    - y: labels for input
    - loss: value of the loss softmax loss function
    - dx: Gradient of the loss with respect to x

    """

    ## calculate softmax probabilities, substract max for numerical stability
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)

    N = x.shape[0]

    ## loss function is negative log-likelihood
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N

    ## from math derivatives we get that for correct class dx -= 1
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx



