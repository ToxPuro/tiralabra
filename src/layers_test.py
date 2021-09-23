"""Tests for layers.py"""

import numpy as np
from layers import *
from helpers import rel_error

def eval_numerical_gradient(f, x, verbose=True, h=0.00001):
    """
    checks numerical gradient by comparing it to f(x+h) - f(x-h) / 2h
    """
  
    grad = np.zeros_like(x)
    # iterate over all indexes in x
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:

        # evaluate function at x+h and x-h
        ix = it.multi_index
        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evalute f(x + h)
        x[ix] = oldval - h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # restore

        # compute the partial derivative with centered formula
        grad[ix] = (fxph - fxmh) / (2 * h)  # the slope
        if verbose:
            print(ix, grad[ix])
        it.iternext()  # step to next dimension

    return grad

def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    same as eval_numerical_gradient but now for arrays
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad





def test_affine_forward():
    """Tests affine_forward against simulated values"""
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                            [ 3.25553199,  3.5141327,   3.77273342]])

    assert rel_error(out, correct_out)<10**-9




def test_affine_backward():
    """Tests affine_backward with gradient checking"""
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    # The error should be around e-10 or less
    print('Testing affine_backward function:')
    assert rel_error(dx_num, dx) < 10**-10
    assert rel_error(dw_num, dw) < 10**-10
    assert rel_error(db_num, db) < 10**-10

def test_relu_forward():
    """Tests relu against simulated values"""
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    out, _ = relu_forward(x)
    correct_out = np.array([[ 0.,          0.,          0.,          0.,        ],
                            [ 0.,          0.,          0.04545455,  0.13636364,],
                            [ 0.22727273,  0.31818182,  0.40909091,  0.5,       ]])


    assert rel_error(out, correct_out) < 5**-8

def test_relu_backward():
    """Tests relu backward with gradient checking"""
    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

    _, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    # The error should be on the order of e-12
    assert rel_error(dx_num, dx) < 4*10**-12

def test_softmax_loss():
    """tests softmax loss against simulated values
    and gradient checking"""
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)


    dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    loss, dx = softmax_loss(x, y)
    assert rel_error(loss, 2.3) < 10**-3
    assert rel_error(dx_num, dx) < 10**-8

def test_batchnorm_forward():
    """Tests batchnorm backward with gradient checking"""
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    gamma = np.ones((D3,))
    beta = np.zeros((D3,))

    # Means should be close to zero and stds close to one.
    print('After batch normalization (gamma=1, beta=0)')
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    assert rel_error(a_norm.mean(axis=0),np.zeros((D3,)))<10**-8
    assert rel_error(a_norm.std(axis=0),np.ones((D3,)))<10**-8

    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])

    # Now means should be close to beta and stds close to gamma.
    print('After batch normalization (gamma=', gamma, ', beta=', beta, ')')
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    assert rel_error(a_norm.mean(axis=0),[11.0,12.0,13.0])<10**-8
    assert rel_error(a_norm.std(axis=0),[1.0,2.0,3.0])<10**-8

def test_batchnorm_backward():
    """Tests batchnorm with gradient checking"""
    np.random.seed(231)
    N, D = 4, 5
    x = 5 * np.random.randn(N, D) + 12
    gamma = np.random.randn(D)
    beta = np.random.randn(D)
    dout = np.random.randn(N, D)

    bn_param = {'mode': 'train'}
    fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]
    fg = lambda a: batchnorm_forward(x, a, beta, bn_param)[0]
    fb = lambda b: batchnorm_forward(x, gamma, b, bn_param)[0]

    dx_num = eval_numerical_gradient_array(fx, x, dout)
    da_num = eval_numerical_gradient_array(fg, gamma.copy(), dout)
    db_num = eval_numerical_gradient_array(fb, beta.copy(), dout)

    _, cache = batchnorm_forward(x, gamma, beta, bn_param)
    dx, dgamma, dbeta = batchnorm_backward(dout, cache)

    # You should expect to see relative errors between 1e-13 and 1e-8.
    assert rel_error(dx_num, dx)<10**-8
    assert rel_error(da_num, dgamma)<10**-8
    assert rel_error(db_num, dbeta)<10**-8

def test_conv_forward():
    x_shape = (2, 3, 4, 4)
    w_shape = (3, 3, 4, 4)
    x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)
    w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)
    b = np.linspace(-0.1, 0.2, num=3)

    conv_param = {'stride': 2, 'pad': 1}
    out, _ = conv_forward(x, w, b, conv_param)
    correct_out = np.array([[[[-0.08759809, -0.10987781],
                                [-0.18387192, -0.2109216 ]],
                                [[ 0.21027089,  0.21661097],
                                [ 0.22847626,  0.23004637]],
                                [[ 0.50813986,  0.54309974],
                                [ 0.64082444,  0.67101435]]],
                            [[[-0.98053589, -1.03143541],
                                [-1.19128892, -1.24695841]],
                                [[ 0.69108355,  0.66880383],
                                [ 0.59480972,  0.56776003]],
                                [[ 2.36270298,  2.36904306],
                                [ 2.38090835,  2.38247847]]]])

    # Compare your output to ours; difference should be around e-8
    print('Testing conv_forward_naive')
    assert rel_error(out, correct_out)<10**-7
