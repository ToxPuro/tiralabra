import numpy as np

def reshape(x):
    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    return x.reshape(row_dim, col_dim)