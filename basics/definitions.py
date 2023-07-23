import dataclasses
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union, Callable

import jax.numpy as jnp

@dataclasses.dataclass
class GPCache:
  """Caching intermediate results for GP."""
  chol: jnp.array
  kinvy: jnp.array
  needs_update: bool


class SubDataset(NamedTuple):
  """Sub dataset with x: n x d and y: n x m; d, m>=1."""
  x: jnp.ndarray
  y: jnp.ndarray
  aligned: Optional[Union[int, str, bool, Tuple[str, ...]]] = None


@dataclasses.dataclass
class GPParams:
  """Parameters in a GP."""

  config: Dict[str, Any] = dataclasses.field(default_factory=lambda: {})
  model: Dict[str, Any] = dataclasses.field(default_factory=lambda: {})
  cache: Dict[Union[int, str], GPCache] = dataclasses.field(
      default_factory=lambda: {}
  )
  samples: List[Dict[str, Any]] = dataclasses.field(default_factory=lambda: [])


AllowedDatasetTypes = Union[
    List[Union[Tuple[jnp.ndarray, ...], SubDataset]],
    Dict[Union[str, int], Union[Tuple[jnp.ndarray, ...], SubDataset]],
]

WarpFuncType = Optional[Dict[str, Callable[[Any], Any]]]
