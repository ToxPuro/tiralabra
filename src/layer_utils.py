from layers import *


def affine_relu_forward(x, w, b):
    """affine + ReLU
    """
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache

def affine_relu_backward(dout, cache):
    """Backward pass for affine + ReLU
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db

def conv_relu_forward(x, w, b, conv_param):
    """convolution + ReLU
    """
    a, conv_cache = conv_forward(x, w, b, conv_param)
    out, relu_cache = relu_forward(a)
    cache = (conv_cache, relu_cache)
    return out, cache


def conv_relu_backward(dout, cache):
    """Backward pass for the conv-relu
    """
    conv_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = conv_backward(da, conv_cache)
    return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
    """convolution + ReLU + maxpool.
    """
    a, conv_cache = conv_forward(x, w, b, conv_param)
    s, relu_cache = relu_forward(a)
    out, pool_cache = max_pool_forward(s, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return out, cache


def conv_relu_pool_backward(dout, cache):
    """Backward pass for convolution + ReLU + maxpool.
    """
    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward(da, conv_cache)
    return dx, dw, db
