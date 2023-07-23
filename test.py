##################################
### Imports and some utilities ###
##################################
#%%
import random
import time
#%%
import jax
import jax.numpy as jnp
#%%
import matplotlib
import matplotlib.pyplot as plt
#%%
from basics import definitions as defs
from gp_utils import gp
from gp_utils import kernel
#%%
font = {
    'family': 'serif',
    'weight': 'normal',
    'size': 7,
}
axes = {'titlesize': 7, 'labelsize': 7}
matplotlib.rc('font', **font)
matplotlib.rc('axes', **axes)

GPParams = defs.GPParams
SubDataset = defs.SubDataset

#%%
def plot_function_samples(
    mean_func,
    cov_func,
    params,
    warp_func=None,
    num_samples=1,
    random_seed=0,
    x_min=0,
    x_max=1,
):
  """Plot function samples from a 1-D Gaussian process."""
  key = jax.random.PRNGKey(random_seed)
  key, y_key = jax.random.split(key, 2)
  x = jnp.linspace(x_min, x_max, 100)[:, None]
  y = gp.sample_from_gp(
      y_key,
      mean_func,
      cov_func,
      params,
      x,
      warp_func=warp_func,
      num_samples=num_samples,
      method='svd',
  )
  fig = plt.figure(dpi=200, figsize=(2, 1))
  plt.plot(x, y)
  plt.xlabel('x')
  plt.ylabel('f(x)')
  
#%%
###########################################################
### Define a ground truth GP and generate training data ###
###########################################################

# @title Define a ground truth GP and generate training data
params = GPParams(
    model={
        'lengthscale': 0.1,
        'signal_variance': 10.0,
        'noise_variance': 1e-6,
        'constant': 5.0,
    }
)  # parameters of the GP

#%%
def ground_truth_mean_func(params, x, warp_func=None):
  return -jax.nn.relu(x - 0.5) * 20

mean_func = ground_truth_mean_func  # mean function of the GP
cov_func = kernel.matern52  # kernel (covariance) function of the GP

random_seed = 10  #@param{type: "number", isTemplate: true}
key = jax.random.PRNGKey(random_seed)
# number of training functions
num_train_functions = 10  #@param{type: "number", isTemplate: true}
# number of datapoints per training function
num_datapoints_per_train_function = 10  #@param{type: "number", isTemplate: true}

#%%
dataset = {}  # Training dataset
# Generate generic training data (only used by NLL)
for sub_dataset_id in range(num_train_functions):
  key, x_key, y_key = jax.random.split(key, 3)
  x = jax.random.uniform(x_key, (num_datapoints_per_train_function, 1))
  y = gp.sample_from_gp(y_key, mean_func, cov_func, params, x, method='svd')
  dataset[str(sub_dataset_id)] = SubDataset(x, y)

#%%  
# Generate matching-input training data (only used by EKL)
key, x_key, y_key = jax.random.split(key, 3)
x = jax.random.uniform(x_key, (num_datapoints_per_train_function, 1))
#%%
y = gp.sample_from_gp(
    y_key,
    mean_func,
    cov_func,
    params,
    x,
    num_samples=num_train_functions,
    method='svd',
)
dataset['matching-input'] = SubDataset(x, y, aligned=1)

#%%
###########################################################
### Visualize function samples from the ground truth GP ###
###########################################################
random_seed = 0  
num_samples = 10  
plot_function_samples(mean_func,
                      cov_func,
                      params,
                      num_samples=num_samples,
                      random_seed=random_seed)
