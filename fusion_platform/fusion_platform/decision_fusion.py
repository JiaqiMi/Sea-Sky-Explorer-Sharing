import numpy as np
from .grid_utils import standardize, minmax_scale

def likelihood_map(measurement_value_map, noise_std):
    """Compute Gaussian likelihood per pixel: p ~ exp(-(m - map)^2 / (2*sigma^2))
    measurement_value_map: dict with keys {'field_name': (meas_value, map2d)}
    Returns dict of likelihood maps (each HxW, normalized to [0,1]).
    """
    like = {}
    for k,(m, fmap) in measurement_value_map.items():
        sigma2 = max(noise_std.get(k, 1.0), 1e-6)**2
        d2 = (fmap - m)**2
        L = np.exp(-0.5 * d2 / sigma2)
        # normalize for display
        L = minmax_scale(L)
        like[k] = L
    return like

def fuse_likelihood(like_dict, weights=None):
    """Multiply likelihoods with optional weights (exponentiation).
    like_dict: {name: Lij}
    weights: dict {name: w}, if None -> equal weights.
    Returns fused posterior (normalized to [0,1]).
    """
    names = list(like_dict.keys())
    if weights is None:
        weights = {n: 1.0 for n in names}
    # log domain sum of weighted logs
    logp = None
    for n in names:
        L = np.clip(like_dict[n], 1e-9, 1.0)
        w = float(weights.get(n, 1.0))
        lp = w * np.log(L)
        logp = lp if logp is None else (logp + lp)
    P = np.exp(logp - np.max(logp))
    P = P / (np.sum(P) + 1e-12)
    # scale to [0,1] for display
    P_disp = (P - P.min())/(P.max()-P.min()+1e-12)
    return P, P_disp
