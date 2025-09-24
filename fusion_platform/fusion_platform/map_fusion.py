import numpy as np
from .grid_utils import regrid_to, standardize

def stack_maps(fields, target_shape=None, standardize_each=True):
    """Align and stack a list of 2D arrays into a 3D stack [H,W,C].
    - target_shape: tuple(H,W). If None, use the first field's shape.
    - standardize_each: z-score each channel to mitigate unit differences.
    Returns stack, info dict.
    """
    assert len(fields) >= 1
    if target_shape is None:
        target_shape = fields[0].shape
    chs = []
    for f in fields:
        g = regrid_to(f, target_shape, method='bilinear')
        if standardize_each:
            g = standardize(g)
        chs.append(g)
    stack = np.stack(chs, axis=-1)  # H, W, C
    return stack, {'target_shape': target_shape, 'channels': len(fields)}

def fuse_map(fields, weights=None, target_shape=None, standardize_each=True):
    """Variance/weight-based linear fusion after stacking.
    - weights: list len=C; if None, use equal weights.
    Returns fused_map(H,W) and stack(H,W,C).
    """
    stack, info = stack_maps(fields, target_shape, standardize_each)
    H, W, C = stack.shape
    if weights is None:
        weights = np.ones(C, dtype=float) / C
    w = np.asarray(weights, dtype=float).reshape(1,1,C)
    fused = np.nansum(stack * w, axis=-1)
    return fused, stack
