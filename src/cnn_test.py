import numpy as np
from helpers import eval_numerical_gradient, eval_numerical_gradient_array, rel_error
from cnn import ThreeLayerConvNet
"""Test for neural net"""

def test_neural_net():
    """Test that gradient backpropagates properly"""
    num_inputs = 2
    input_dim = (3, 16, 16)
    reg = 0.0
    num_classes = 10
    np.random.seed(231)
    X = np.random.randn(num_inputs, *input_dim)
    y = np.random.randint(num_classes, size=num_inputs)

    model = ThreeLayerConvNet(
        num_filters=3,
        filter_size=3,
        input_dim=input_dim,
        hidden_dim=7,
        dtype=np.float64
    )
    loss, grads = model.loss(X, y)
    for param_name in sorted(grads):
        f = lambda _: model.loss(X, y)[0]
        param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)
        assert rel_error(param_grad_num, grads[param_name])<10**-1
