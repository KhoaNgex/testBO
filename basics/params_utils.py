from typing import Any, Callable, Dict, List, Optional, Union

from basics import definitions as defs

GPParams = defs.GPParams

def _verify_params(model_params: Dict[str, Any], expected_keys: List[str]):
    """Verify that dictionary params has the expected keys."""
    if not set(expected_keys).issubset(set(model_params.keys())):
        raise ValueError(f'Expected parameters are {sorted(expected_keys)}, '
                        f'but received {sorted(model_params.keys())}.')

def retrieve_params(
    params: GPParams,
    keys: List[str],
    warp_func: Optional[Dict[str, Callable[[Any], Any]]] = None) -> List[Any]:
    """Returns a list of parameter values (warped if specified) by keys' order."""
    model_params = params.model
    _verify_params(model_params, keys)
    if warp_func:
        values = [
            warp_func[key](model_params[key])
            if key in warp_func else model_params[key] for key in keys
        ]
    else:
        values = [model_params[key] for key in keys]
    return values