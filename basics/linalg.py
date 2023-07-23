import jax
from jax.custom_derivatives import custom_vjp
import jax.numpy as jnp

# An implementation of sqrt that returns a large value for the gradient at 0
# instead of nan.
_safe_sqrt = jax.custom_vjp(jnp.sqrt)

def _safe_sqrt_fwd(x):
  result, vjpfun = jax.vjp(jnp.sqrt, x)
  return result, (x, vjpfun)

def _safe_sqrt_rev(primals, tangent):
  x, vjpfun = primals
  # max_grad = dtypes.finfo(dtypes.dtype(x)).max
  max_grad = 1e6
  result = jnp.where(x != 0., vjpfun(tangent)[0], jnp.full_like(x, max_grad))
  return (result,)


_safe_sqrt.defvjp(_safe_sqrt_fwd, _safe_sqrt_rev)


def safe_l2norm(x):
  """Safe way to compute l2 norm on x without a nan gradient."""
  sqdist = jnp.sum(x**2)
  return _safe_sqrt(sqdist)