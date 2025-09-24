import numpy as np

def regrid_to(src, target_shape, method='bilinear'):
    """Resample 2D array to target_shape using simple methods (nearest/bilinear).
    Ultra-lightweight, no geo-awareness (treats as image resampling).
    """
    src = np.asarray(src, dtype=float)
    H, W = src.shape
    th, tw = target_shape
    if H == th and W == tw:
        return src.copy()
    # normalized grid
    y = np.linspace(0, H-1, th)
    x = np.linspace(0, W-1, tw)
    yy, xx = np.meshgrid(y, x, indexing='ij')
    if method == 'nearest':
        yi = np.rint(yy).astype(int)
        xi = np.rint(xx).astype(int)
        yi = np.clip(yi, 0, H-1)
        xi = np.clip(xi, 0, W-1)
        return src[yi, xi]
    # bilinear
    y0 = np.floor(yy).astype(int); x0 = np.floor(xx).astype(int)
    y1 = np.clip(y0+1, 0, H-1);    x1 = np.clip(x0+1, 0, W-1)
    wy = yy - y0; wx = xx - x0
    Ia = src[y0, x0]; Ib = src[y0, x1]; Ic = src[y1, x0]; Id = src[y1, x1]
    return (Ia*(1-wx)*(1-wy) + Ib*(wx)*(1-wy) + Ic*(1-wx)*wy + Id*wx*wy)

def standardize(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=float)
    m = np.nanmean(arr); s = np.nanstd(arr)
    return (arr - m) / (s + eps)

def minmax_scale(arr, eps=1e-8):
    arr = np.asarray(arr, dtype=float)
    mn = np.nanmin(arr); mx = np.nanmax(arr)
    return (arr - mn) / (mx - mn + eps)
