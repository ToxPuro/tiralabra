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