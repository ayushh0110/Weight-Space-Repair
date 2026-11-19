"""Weight-Space Repair.

Given a failing prompt and a model, find out what failed and why. Causes outside
the weights (file, config, request, sampling, runtime) get an exact fix; only a
failure that survives all of those is called weight-level, diagnosed inside the
network and, where a checked fix exists, repaired (a weight patch, a steering or
serving settings).
"""
import os as _os
import sys as _sys

__version__ = "0.1.0"

# On a Mac, PyTorch's MPS backend lacks some ops (an SVD, for one); with this set they run on the CPU instead of
# failing. PyTorch reads it when torch is first imported, so it is set here, before any wsr module imports torch,
# and only when the user has not set it. It changes nothing for CPU or CUDA runs.
if _sys.platform == "darwin" and "torch" not in _sys.modules:
    _os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
