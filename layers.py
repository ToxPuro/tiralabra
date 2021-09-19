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

def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    idea of batchnorm is to keep the earlier layers distribution more stable while the weights are changing
    to do this we normalize the data. Also we include gamma and beta that the neural net can if it wants to forget the normalization if that is better
    (https://arxiv.org/abs/1502.03167) 

    - x: input data
    - gamma: Scale parameter
    - beta: Shift paremeter
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test' in order to know how to act
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: running_mean used after training
      - running_var: running_var used after training

    - out: normalized output
    - cache: cache values for backprop
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    
    if mode == "train":
        ## during training do normal normalization
        sample_mean = x.mean(axis=0)
        sample_var = x.var(axis=0) + eps
        sample_std = np.sqrt(sample_var)

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        z = (x - sample_mean)/sample_std
        out = z*gamma + beta
        cache = (x,sample_mean,sample_std,gamma,beta,z,sample_var)

    elif mode == "test":
        ## during testing use the calculated running mean and variance to normalize
        z = (x - running_mean)/np.sqrt(running_var+eps)
        out = z*gamma + beta


    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache



