""""Implementations of different update rules for neural"""

import numpy as np




def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    In order to decrease the loss function we need to go in the direction of negative gradient

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)

    w -= config["learning_rate"] * dw
    return w, config

def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    With momentum added the neural net will keep on going with momentum so noise in the gradient doesn't
    affect as much and we don't get stuck in local minima

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("momentum", 0.9)
    v = config.get("velocity", np.zeros_like(w))

    v = config["momentum"] * v - config["learning_rate"] * dw
    next_w = w+v
    config["velocity"] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    read more from here: https://towardsdatascience.com/understanding-rmsprop-faster-neural-network-learning-62e116fcf29a

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-2)
    config.setdefault("decay_rate", 0.99)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("cache", np.zeros_like(w))

    config["cache"] = config["decay_rate"]*config["cache"]+(1-config["decay_rate"])*dw**2
    next_w = w - config["learning_rate"]*dw /(np.sqrt(config["cache"]) + config["epsilon"])

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    Read more from here https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault("learning_rate", 1e-3)
    config.setdefault("beta1", 0.9)
    config.setdefault("beta2", 0.999)
    config.setdefault("epsilon", 1e-8)
    config.setdefault("m", np.zeros_like(w))
    config.setdefault("v", np.zeros_like(w))
    config.setdefault("t", 0)

    eps, learning_rate = config['epsilon'], config['learning_rate']
    beta1, beta2 = config['beta1'], config['beta2']
    m, v, t = config['m'], config['v'], config['t']
    # Adam
    t = t + 1
    m = beta1 * m + (1 - beta1) * dw          # momentum
    mt = m / (1 - beta1**t)                   # bias correction
    v = beta2 * v + (1 - beta2) * (dw * dw)   # RMSprop
    vt = v / (1 - beta2**t)                   # bias correction
    next_w = w - learning_rate * mt / (np.sqrt(vt) + eps)
    # update values
    config['m'], config['v'], config['t'] = m, v, t

    return next_w, config
