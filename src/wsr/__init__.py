"""Weight-Space Repair.

Given a failing prompt and a model, find out what failed and why. Causes outside
the weights (file, config, request, sampling, runtime) get an exact fix; only a
failure that survives all of those is called weight-level, diagnosed inside the
network and, where a checked fix exists, repaired (a weight patch, a steering or
serving settings).
"""

__version__ = "0.1.0"
