import numpy as np
def sum_spectral_channels(cube): return cube.sum(axis=2,dtype=np.uint64)
