"""Weight-Space Repair: triage layer.

Given a failing prompt and a model, find out what failed and why. Causes outside
the weights (file, config, request, sampling, runtime) get an exact fix; only a
failure that survives all of those is called weight-level.
"""

__version__ = "0.1.0"
