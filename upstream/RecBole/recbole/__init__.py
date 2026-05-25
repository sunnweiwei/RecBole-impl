from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

__version__ = "1.2.1"

# Monkey patch torch.load for PyTorch 2.6+ backward compatibility
import inspect
import torch
_original_load = torch.load

if 'weights_only' in inspect.signature(_original_load).parameters:
    def _safe_load(f, map_location=None, pickle_module=None, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)
    torch.load = _safe_load

