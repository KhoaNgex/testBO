import functools
import jax
import jax.numpy as jnp

from basics import params_utils
from basics import linalg

retrieve_params = params_utils.retrieve_params
vmap = jax.vmap

def covariance_matrix(kernel):
  @functools.wraps(kernel)
  def matrix_map(params, vx1, vx2=None, warp_func=None, diag=False):
    """Returns the kernel matrix of input array vx1 and input array vx2."""
    cov_func = functools.partial(kernel, params, warp_func=warp_func)
    mmap = vmap(lambda x: vmap(lambda y: cov_func(x, y))(vx1))
    if vx2 is None:
      if diag:
        return vmap(lambda x: cov_func(x, x))(vx1)
      vx2 = vx1
    return mmap(vx2).T

  return matrix_map

@covariance_matrix
def matern52(params, x1, x2, warp_func=None):
  """Matern 5/2 kernel: Eq.(4.17) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r = jnp.sqrt(5) * linalg.safe_l2norm((x1 - x2) / lengthscale)
  return signal_variance * (1 + r + r**2 / 3) * jnp.exp(-r)