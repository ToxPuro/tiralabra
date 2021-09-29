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

def affine_norm_relu_forward(x,w,b,gamma,beta,bn_param, normalization, dropout, do_param):
    """"affine + (norm) + ReLU + (dropout)
    """
    bn_cache, do_cache = None, None
    #affine layer
    out, fc_cache = affine_forward(x,w,b)
    if normalization == "batchnorm":
        out, bn_cache = batchnorm_forward(out, gamma, beta, bn_param)
    elif normalization == "layernorm":
        out, bn_cache = layernorm_forward(out, gamma, beta, bn_param)
    #relu
    out, relu_cache = relu_forward(out)
    #dropout
    if dropout:
        out, do_cache = dropout_forward(out, do_param)
    return out, (fc_cache, bn_cache, relu_cache, do_cache)

def affine_norm_relu_backward(dout, cache, normalization, dropout):
    """"backward pass for affine + (norm) + ReLU + (dropout)
    """
    fc_cache, bn_cache, relu_cache, do_cache = cache
    if dropout:
        dout = dropout_backward(dout, do_cache)
    
    dout = relu_backward(dout, relu_cache)
    
    dgamma, dbeta = None, None

    if normalization == "batchnorm":
        dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache)
    elif normalization == "layernorm":
        dout, dgamma, dbeta = layernorm_backward(dout, bn_cache)
    
    dx, dw, db = affine_backward(dout, fc_cache)
    return dx, dw, db, dgamma, dbeta

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


def conv_bn_relu_forward(x, w, b, gamma, beta, conv_param, bn_param):
    """convolution + batchnorm + ReLU
    """
    a, conv_cache = conv_forward(x, w, b, conv_param)
    an, bn_cache = spatial_batchnorm_forward(a, gamma, beta, bn_param)
    out, relu_cache = relu_forward(an)
    cache = (conv_cache, bn_cache, relu_cache)
    return out, cache


def conv_bn_relu_backward(dout, cache):
    """Backward pass for convolution + batchnorm + ReLU.
    """
    conv_cache, bn_cache, relu_cache = cache
    dan = relu_backward(dout, relu_cache)
    da, dgamma, dbeta = spatial_batchnorm_backward(dan, bn_cache)
    dx, dw, db = conv_backward(da, conv_cache)
    return dx, dw, db, dgamma, dbeta


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
