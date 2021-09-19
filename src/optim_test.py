from optim import *
from helpers import *

def test_sgd():
  N, D = 4, 5
  w = np.linspace(-0.4, 0.6, num=N*D).reshape(N, D)
  dw = np.linspace(-0.6, 0.4, num=N*D).reshape(N, D)

  config = {"learning_rate": 1e-3}
  next_w, _ = sgd(w, dw)

  expected_next_w = np.asarray([
    [ 0.1406,      0.20738947,  0.27417895,  0.34096842,  0.40775789],
    [ 0.47454737,  0.54133684,  0.60812632,  0.67491579,  0.74170526],
    [ 0.80849474,  0.87528421,  0.94207368,  1.00886316,  1.07565263],
    [ 1.14244211,  1.20923158,  1.27602105,  1.34281053,  1.4096    ]])

  # Should see relative errors around e-8 or less
  rel_error(next_w, expected_next_w)<10**-8