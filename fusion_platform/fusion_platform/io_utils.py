import numpy as np

def load_field(path_or_array):
    """Load a field as numpy array. If input is already an array, pass-through.
    GeoTIFF read is attempted if rasterio is present; otherwise fallback to np.load for .npy.
    Returns (array, metadata_dict). Metadata is minimal unless rasterio is available.
    """
    if isinstance(path_or_array, np.ndarray):
        return path_or_array.astype(float), {'crs': None, 'transform': None}
    p = str(path_or_array)
    try:
        import rasterio
        with rasterio.open(p) as ds:
            arr = ds.read(1).astype(float)
            meta = {'crs': ds.crs, 'transform': ds.transform, 'bounds': ds.bounds}
            return arr, meta
    except Exception:
        # fallback simple loaders
        if p.lower().endswith('.npy'):
            arr = np.load(p).astype(float)
            return arr, {'crs': None, 'transform': None}
        # very simple ASCII reader (rows of numbers)
        try:
            arr = np.loadtxt(p).astype(float)
            return arr, {'crs': None, 'transform': None}
        except Exception:
            raise RuntimeError(f"Unsupported file or missing rasterio: {p}")

def save_array_as_image(arr, path):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(arr)
    plt.title(path.split('/')[-1])
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
