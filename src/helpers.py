import numpy as np

def reshape(x):
    row_dim = x.shape[0]
    col_dim = np.prod(x.shape[1:])
    return x.reshape(row_dim, col_dim)

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))