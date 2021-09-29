import numpy as np
from layers import softmax_loss
from layer_utils import *


class ThreeLayerConvNet():
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(1, 28, 28),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        C, H, W = input_dim

        # conv layer weights - using the channel-first format of the input
        self.params['W1'] = np.random.randn(num_filters, C, filter_size, filter_size) * weight_scale        

        # number of biases = number of filters
        self.params['b1'] = np.zeros(num_filters)


        # maxpool output volume dims
        # maxpool receives original input image dims as input since inputs dims are preserved
        HP, WP = (H - 2)//2 + 1, (W - 2)//2 + 1  # 2x2 max pooling assuming typical S = 2
        
        # hidden affine layer params
        self.params['W2'] = np.random.randn(num_filters * HP * WP, hidden_dim) * weight_scale
        self.params['b2'] = np.zeros(hidden_dim)

        # output affine layer params
        self.params['W3'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b3'] = np.zeros(num_classes)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the neural net
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        scores = None

        # (1) conv - relu - 2x2 max pool
        pool_out, conv_cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)

        # (2) affine (FC) - relu
        A2, fc1_cache = affine_relu_forward(pool_out, W2, b2)

        # (3) affine (FC) - softmax 
        scores, fc2_cache = affine_forward(A2, W3, b3)

        if y is None:
            return scores

        loss, grads = 0, {}

        # (3) softmax 
        # calculate data loss
        loss, softmax_grad = softmax_loss(scores, y)

        # add regularization loss for all weights
        loss += 0.5 * self.reg * np.sum(W1 * W1)
        loss += 0.5 * self.reg * np.sum(W2 * W2)
        loss += 0.5 * self.reg * np.sum(W3 * W3)
        
        # (3) affine (FC)
        dout, grads['W3'], grads['b3'] = affine_backward(softmax_grad, fc2_cache)

        # (2) affine (FC) - relu
        dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, fc1_cache)

        # (1) conv - relu - 2x2 max pool
        dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, conv_cache)

        # L2 regularization
        grads['W1'] += self.reg * W1
        grads['W2'] += self.reg * W2
        grads['W3'] += self.reg * W3

        return loss, grads