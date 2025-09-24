import numpy as np
from .grid_utils import regrid_to, standardize, minmax_scale

SOBEL_X = np.array([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=float)
SOBEL_Y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=float)

def conv2(a, k):
    H,W = a.shape; kh, kw = k.shape
    ph, pw = kh//2, kw//2
    ap = np.pad(a, ((ph,ph),(pw,pw)), mode='edge')
    out = np.zeros_like(a, dtype=float)
    for i in range(H):
        for j in range(W):
            out[i,j] = np.sum(ap[i:i+kh, j:j+kw] * k)
    return out

def gradient_mag(a):
    gx = conv2(a, SOBEL_X)
    gy = conv2(a, SOBEL_Y)
    return np.hypot(gx, gy)

def local_std(a, win=7):
    H,W = a.shape; r = win//2
    ap = np.pad(a, ((r,r),(r,r)), mode='reflect')
    out = np.zeros_like(a, dtype=float)
    # integral image for mean and mean of squares
    ii  = ap.cumsum(0).cumsum(1)
    ii2 = (ap*ap).cumsum(0).cumsum(1)
    def rect_sum(ii, y0,x0,y1,x1):
        s = ii[y1,x1]
        if y0>0: s -= ii[y0-1,x1]
        if x0>0: s -= ii[y1,x0-1]
        if y0>0 and x0>0: s += ii[y0-1,x0-1]
        return s
    for i in range(H):
        for j in range(W):
            y0=i; x0=j; y1=i+2*r; x1=j+2*r
            area = (2*r+1)*(2*r+1)
            s1 = rect_sum(ii, y0,x0,y1,x1)
            s2 = rect_sum(ii2,y0,x0,y1,x1)
            m = s1/area
            v = max(s2/area - m*m, 0.0)
            out[i,j] = np.sqrt(v)
    return out

def build_feature_tensor(fields, target_shape=None, feature_set=('grad','rough'), standardize_each=True):
    """Compute per-field features and stack to [H,W,Cfeat].
    feature_set options:
      - 'grad'  : Sobel gradient magnitude
      - 'rough' : local std (roughness)
      - 'raw'   : raw (standardized) field
    """
    base = []
    for f in fields:
        g = f if (target_shape is None) else regrid_to(f, target_shape, 'bilinear')
        if standardize_each:
            g = standardize(g)
        chs = []
        for name in feature_set:
            if name == 'grad':
                chs.append(standardize(gradient_mag(g)))
            elif name == 'rough':
                chs.append(standardize(local_std(g, win=7)))
            elif name == 'raw':
                chs.append(g)
        base.append(chs)  # list of channels for this field
    # concatenate across fields
    feat_maps = [c for per_field in base for c in per_field]
    tensor = np.stack(feat_maps, axis=-1)  # H,W,C
    return tensor

def pca_fusion(tensor, n_components=1, eps=1e-8):
    """Very light PCA over channels to get principal fused map(s).
    Returns fused_list (each HxW) and components (C x k) for interpretability.
    """
    H,W,C = tensor.shape
    X = tensor.reshape(-1, C)
    # center
    mu = X.mean(0, keepdims=True)
    Xc = X - mu
    # SVD
    U,S,VT = np.linalg.svd(Xc, full_matrices=False)
    comps = VT[:n_components]  # (k, C)
    fused = (Xc @ comps.T).reshape(H,W,n_components)
    return [fused[...,i] for i in range(n_components)], comps
