import jax
import jax.random

from basics import params_utils
retrieve_params = params_utils.retrieve_params

def sample_from_gp(key,
                   mean_func,
                   cov_func,
                   params,
                   x,
                   warp_func=None,
                   num_samples=1,
                   method='cholesky',
                   eps=1e-6):
    """Sample a function from a GP and return its evaluations on x (n x d)."""
    mean = mean_func(params, x, warp_func=warp_func)
    noise_variance, = retrieve_params(
        params, ['noise_variance'], warp_func=warp_func)
    cov = cov_func(params, x, warp_func=warp_func)
    return (jax.random.multivariate_normal(
        key,
        mean.flatten(),
        cov + jax.numpy.eye(len(x)) * (noise_variance + eps),
        shape=(num_samples,),
        method=method)).T
