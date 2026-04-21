import warnings

try:
    import cupy as xp
    _test_array = xp.array([1, 2, 3])
    USE_GPU = True
    print("🚀 [System] GPU detected, CuPy hardware acceleration automatically enabled!")
except (ImportError, Exception):
    import numpy as xp
    USE_GPU = False
    # print("🐢 [System] No available GPU detected, falling back to NumPy (CPU).")

__all__ = ['xp', 'USE_GPU']
