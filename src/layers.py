"""Different kinds of layers for the neural net"""

import numpy as np
from helpers import reshape, im2col, col2im


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
    cache = (x, w)
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
    x, w = cache

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

    idea of batchnorm is to keep the earlier layers distribution
    more stable while the weights are changing
    to do this we normalize the data. Also we include
    gamma and beta that the neural net can
    if it wants to forget the normalization if that is better
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

    _, D = x.shape
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
        cache = (sample_std,gamma,z)

    elif mode == "test":
        ## during testing use the calculated running mean and variance to normalize
        z = (x - running_mean)/np.sqrt(running_var+eps)
        out = z*gamma + beta


    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache

def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.
    Reference: (https://arxiv.org/abs/1502.03167)

    -dout: upstream gradient
    -cache: cached values form batchnorm_forward

    -dx: gradient in respect to x
    -dgamma: gradient in respect to gamma
    -dbeta: gradient in respect to beta
    """

    sample_std,gamma,z = cache

    dbeta = dout.sum(axis=0)
    dgamma = np.sum(dout * z, axis=0)



    N = dout.shape[0]
    dfdz = dout * gamma
    dfdz_sum = np.sum(dfdz,axis=0)
    dx = dfdz - dfdz_sum/N - np.sum(dfdz * z,axis=0) * z/N
    dx /= sample_std

    return dx, dgamma, dbeta

def conv_forward(x,w,b,conv_param):
    """
        Performs a forward convolution.

        Parameters:
        - X : Last conv layer of shape (m, n_C_prev, n_H_prev, n_W_prev).
        Returns:
        - out: previous layer convolved.
    """

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    ## get dimensions
    m, C_prev, n_H_prev, n_W_prev = x.shape
    F, C, HF, WF = w.shape


    ## calculate dimensions after convolutions
    n_H = int((n_H_prev + 2 * pad - HF)/ stride) + 1
    n_W = int((n_W_prev + 2 * pad - WF)/ stride) + 1


    ## image into matrix
    x_col = im2col(x, HF, WF, stride, pad)

    ##flatten filters
    w_col = w.reshape((F, -1))
    b_col = b.reshape(-1, 1)
    # Perform matrix multiplication.
    out = w_col @ x_col + b_col
    # Reshape back matrix to image.
    out = np.array(np.hsplit(out, m)).reshape((m, F, n_H, n_W))
    cache = (x, x_col, w_col, w, conv_param)
    return out, cache

def conv_backward(dout, cache):
    """
        Distributes error from previous layer to convolutional layer and
        compute error for the current convolutional layer.
        Parameters:
        - dout: error from previous layer.

        Returns:
        - dX: error of the current convolutional layer.
        - self.W['grad']: weights gradient.
        - self.b['grad']: bias gradient.
    """

    x, x_col, w_col, w, conv_param = cache

    stride = conv_param["stride"]
    pad = conv_param["pad"]

    F, C, HH, WW = w.shape

    m, _, _, _ = x.shape
    # Compute bias gradient.
    db = np.sum(dout, axis=(0,2,3))
    # Reshape dout properly.
    dout = dout.reshape(dout.shape[0] * dout.shape[1], dout.shape[2] * dout.shape[3])
    dout = np.array(np.vsplit(dout, m))
    dout = np.concatenate(dout, axis=-1)
    # Perform matrix multiplication between reshaped dout and w_col to get dX_col.
    dx_col = w_col.T @ dout
    # Perform matrix multiplication between reshaped dout and X_col to get dW_col.
    dw_col = dout @ x_col.T
    # Reshape back to image (col2im).
    dx = col2im(dx_col, x.shape, HH, WW, stride, pad)
    # Reshape dw_col into dw.
    dw = dw_col.reshape((F, C, HH, WW))

    return dx, dw, db

def max_pool_forward(x, pool_param):
    """
    A fast implementation of the forward pass for a max pooling layer.

    This chooses between the reshape method and the naive method. If the pooling
    regions are square and tile the input image, then we can use the reshape
    method which is very fast. Otherwise we fall back on the naive method, which
    is clearly slower.
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]

    same_size = pool_height == pool_width == stride
    tiles = H % pool_height == 0 and W % pool_width == 0
    if same_size and tiles:
        out, reshape_cache = max_pool_forward_reshape(x, pool_param)
        cache = ("reshape", reshape_cache)
    else:
        out, naive_cache = max_pool_forward_naive(x, pool_param)
        cache = ("naive", naive_cache)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A fast implementation of the backward pass for a max pooling layer.

    This switches between the reshape method an the im2col method depending on
    which method was used to generate the cache.
    """
    
    method, real_cache = cache
    if method == "reshape":
        return max_pool_backward_reshape(dout, real_cache)

    return max_pool_backward_naive(dout, real_cache)



def max_pool_forward_reshape(x, pool_param):
    """
    Max pooling technique which uses square filters that tile the image
    Main idea is to fold the image and then take multidimensional max
    """
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param["pool_height"], pool_param["pool_width"]
    stride = pool_param["stride"]
    assert pool_height == pool_width == stride, "Invalid pool params"
    assert H % pool_height == 0
    assert W % pool_height == 0

    ##This is somewhat tricky so it is somewhat in order to explain the code. At least it was tricky to code
    ## Thing only to doing max_pooling to a single channel of a single image, so N and C stay the same
    ## Now we got a 2-dimensional image. The pooling filter will go tile exactly, this is why to have two implementations,
    ## You can first imagine folding the 2-dimensional image over the height of the filter. So 9x9 image will become 3x9x3 image with a 3x3 filter.
    ## Now the filter needs to only move horizontally and calculate also with the new dimension
    ## Now fold this 3x9x3 by the width and you get 3x3x3x3 image which is sadly impossible to visualize :(
    ## Now the folded image is the same as filter added two new dimensions. When we take the maximum along these dimensions we get the same maximums as sliding the filter :DD

    x_reshaped = x.reshape(
        N, C, H // pool_height, pool_height, W // pool_width, pool_width
    )
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache


def max_pool_backward_reshape(dout, cache):
    """
    Can be only used if used folding technique to in forward
    Input:
        -dout: upstream gradient
        -cached: cached values
    Returns:
        -dx: gradient in respect to x
    """
    x, x_reshaped, out = cache

    ## make dx and out similar to folded image
    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    ##select max pixels
    mask = x_reshaped == out_newaxis
    ## make upstream gradient similar to folded image
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    ## gradient only for max values
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    ##reshape into image shape
    dx = dx_reshaped.reshape(x.shape)

    return dx



def max_pool_forward_naive(x, pool_param):
    """forward pass for a max-pooling layer.
    in max-pooling-layer the images are downsampled because
    otherwise after doing convolutions after convolutions the channel depth of our images gets out of hand
    we only keep the maximum values of convolution areas visualization here:
    https://towardsdatascience.com/lets-code-convolutional-neural-network-in-plain-numpy-ce48e732f5d5

    - x: Input data
    - pool_param: tells us how to max pool includes pooling height, wight and stride

    returns downsampled images

    """
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)
    Hout = (H - HH) // stride + 1
    Wout = (W - WW) // stride + 1


    out = np.zeros((N, C, Hout, Wout))


    for n in range(N): # for each neuron
        for i in range(Hout): # for each y activation
            for j in range(Wout): # for each x activation
                out[n, :, i, j] = np.amax(x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW], axis=(-1, -2))

    cache = (x, pool_param)
    return out, cache

def max_pool_backward_naive(dout, cache):
    """backward pass for a max-pooling layer.
    """
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    Hout = (H - HH) // stride + 1
    Wout = (W - WW) // stride + 1

    dx = np.zeros_like(x)

    for n in range(N): # for each neuron
        for c in range(C): # for each channel
            for i in range(Hout): # for each y activation
                for j in range(Wout): # for each x activation
                    # pass gradient only through indices of max pool
                    ind = np.argmax(x[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW])
                    ind1, ind2 = np.unravel_index(ind, (HH, WW))
                    dx[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW][ind1, ind2] = dout[n, c, i, j]
    return dx
