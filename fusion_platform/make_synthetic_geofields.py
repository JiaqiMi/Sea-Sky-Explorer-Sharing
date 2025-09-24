#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
make_synthetic_geofields.py
生成三张可用于 Geo Fusion IDE 的合成地图：
- 地形（海底地形/测深，单位 m，负值为更深）
- 重力自由空气异常（单位 mGal）
- 地磁异常（单位 nT）

特点：
- 统一网格与分辨率（默认 512x512，像元 ~500 m，可配置）
- 统计参数贴近真实海洋场景（量级、纹理、相关性）
- 同时输出 .npy（IDE 直接读取）、.png 快览，若安装 rasterio 还会输出 GeoTIFF

运行：
    python make_synthetic_geofields.py --H 512 --W 512 --dx_km 0.5 --seed 42

输出目录：synthetic_fields_out/
"""
import os, argparse, json, numpy as np

try:
    import rasterio
    from rasterio.transform import from_origin
    HAS_RIO = True
except Exception:
    HAS_RIO = False

def mk_coords(H, W, dx_km=0.5):
    """构造等间距笛卡尔网格坐标（单位 km）"""
    x = (np.arange(W) - W/2) * dx_km
    y = (np.arange(H) - H/2) * dx_km
    xx, yy = np.meshgrid(x, y, indexing='xy')
    return xx, yy, x, y

def rand_gaussians(xx, yy, centers, amps, sigmas):
    """多高斯叠加"""
    z = np.zeros_like(xx, dtype=float)
    for (cx, cy), a, s in zip(centers, amps, sigmas):
        z += a * np.exp(-(((xx-cx)**2 + (yy-cy)**2) / (2*s*s)))
    return z

def spectral_fractal(H, W, beta=2.0, scale=1.0, seed=0):
    """生成 1/f^beta 频谱噪声（各向同性），返回零均值场"""
    rng = np.random.default_rng(seed)
    # 频率网格
    kx = np.fft.fftfreq(W)
    ky = np.fft.fftfreq(H)
    kxx, kyy = np.meshgrid(kx, ky, indexing='xy')
    k2 = kxx**2 + kyy**2
    k2[0,0] = 1.0  # 避免除零（DC）
    amp = 1.0 / (k2**(beta/2.0))
    phase = rng.uniform(0, 2*np.pi, size=(H,W))
    spec = amp * (np.cos(phase) + 1j*np.sin(phase))
    f = np.fft.ifft2(spec).real
    f = (f - f.mean()) / (f.std() + 1e-8)
    return scale * f

def highpass(arr, sigma_pix=8):
    """简单高通：arr - 高斯模糊"""
    from scipy.ndimage import gaussian_filter
    return arr - gaussian_filter(arr, sigma=sigma_pix)

def rotate_coords(xx, yy, angle_deg):
    """旋转坐标（度）"""
    th = np.deg2rad(angle_deg)
    xr =  np.cos(th)*xx + np.sin(th)*yy
    yr = -np.sin(th)*xx + np.cos(th)*yy
    return xr, yr

def make_bathymetry(H, W, dx_km, seed=0):
    """合成海底地形（m，负值更深）"""
    xx, yy, _, _ = mk_coords(H, W, dx_km)
    rng = np.random.default_rng(seed)

    # 大尺度构造：海盆 + 海山 + 海沟
    centers = [( -30,  20), ( 40, -25), ( 10, 10)]
    amps_m  = [-1200,  900,  600]     # 海沟负值，海山正值（这里最终整体加到负基准上）
    sigmas  = [  25,   18,   12]      # km

    z_long = rand_gaussians(xx, yy, centers, amps_m, sigmas)

    # 基准深度与缓坡（远洋 -3500 ~ -5000 m）
    base = -4200.0 + 0.8*yy  # 南北向缓坡，幅度 ~ 数百米

    # 中小尺度粗糙度（分形）
    rough = spectral_fractal(H, W, beta=2.2, scale=300.0, seed=seed+1)  # RMS ~300 m

    # 组合并控制范围
    bathy = base + z_long + rough
    # 限幅（避免极端值）
    bathy = np.clip(bathy, -6000, -500)

    return bathy

def make_gravity_freeair(bathy_m, dx_km, seed=0):
    """合成自由空气重力异常（mGal）
    经验：海洋自由空气异常与高通地形正相关；再叠加长波趋势与噪声。
    """
    import scipy.ndimage as nd
    H, W = bathy_m.shape
    rng = np.random.default_rng(seed)

    # 高通地形 -> 重力响应（经验比例系数：~0.02 mGal/m，取决于尺度）
    # 先做中尺度高通（~10 km）
    sigma_pix = max(1, int(10.0 / dx_km / 2.355))  # 把 10 km 转换成像素的sigma
    hp = bathy_m - nd.gaussian_filter(bathy_m, sigma=sigma_pix)
    g_from_bathy = -0.02 * hp  # 海底更深（负）对应负重力异常，取负号

    # 加一个长波趋势（地幔、大地水准面等引起）
    xx, yy, _, _ = mk_coords(H, W, dx_km)
    trend = 20.0*np.sin(2*np.pi*xx/200.0) + 15.0*np.cos(2*np.pi*yy/300.0)  # mGal，200-300 km 尺度

    # 小噪声
    noise = rng.normal(0, 1.5, size=(H,W))  # mGal

    g = g_from_bathy + trend + noise
    # 限幅
    g = np.clip(g, -80, 80)
    return g

def make_magnetic_anomaly(H, W, dx_km, seed=0, stripe_angle_deg=30.0):
    """合成海底地磁异常（nT）
    - 条带状海底扩张磁化带（旋转角度）
    - 叠加多个波数与包络
    - 小尺度噪声
    """
    rng = np.random.default_rng(seed)
    xx, yy, _, _ = mk_coords(H, W, dx_km)
    xr, yr = rotate_coords(xx, yy, stripe_angle_deg)

    # 多频条带（沿 xr 方向变化）
    k_list = [2*np.pi/8.0, 2*np.pi/12.0, 2*np.pi/20.0]  # 周期 ~8/12/20 km
    amp_list = [180, 120, 80]  # 振幅（nT）
    mag = np.zeros_like(xr, dtype=float)
    for k, a in zip(k_list, amp_list):
        mag += a * np.sin(k * xr + rng.uniform(0, 2*np.pi))

    # 包络（随距海岭距离变化），让南北方向上有缓变
    env = 0.5 + 0.5*np.tanh(yr/30.0)  # -∞到+∞上从0到1变化
    mag *= env

    # 增添局部火山体等高频斑块
    spots = spectral_fractal(H, W, beta=1.5, scale=40.0, seed=seed+2)

    # 仪器/环境噪声
    noise = rng.normal(0, 10.0, size=(H,W))

    total = mag + spots + noise
    # 限幅
    total = np.clip(total, -600, 600)
    return total

def save_outputs(bathy, grav, mag, out_dir, dx_km):
    os.makedirs(out_dir, exist_ok=True)
    meta = {
        "grid": {"H": int(bathy.shape[0]), "W": int(bathy.shape[1]), "dx_km": float(dx_km)},
        "units": {"terrain":"m (depth, negative means deeper)",
                  "gravity":"mGal (free-air anomaly)",
                  "magnetic":"nT (total field anomaly)"},
        "notes": "Synthetic fields for Geo Fusion IDE. Values approximate marine magnitudes."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # Save NPY
    np.save(os.path.join(out_dir, "terrain_bathy_m.npy"), bathy)
    np.save(os.path.join(out_dir, "gravity_freeair_mgal.npy"), grav)
    np.save(os.path.join(out_dir, "magnetic_anomaly_nt.npy"), mag)

    # Quicklook PNGs
    try:
        import matplotlib.pyplot as plt
        for arr, name in [(bathy,"terrain_bathy_m"),
                          (grav,"gravity_freeair_mgal"),
                          (mag,"magnetic_anomaly_nt")]:
            plt.figure()
            plt.imshow(arr)
            plt.title(name)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, name + ".png"), dpi=150)
            plt.close()
    except Exception:
        pass

    # Optional GeoTIFF (pixel grid, no projection by default)
    if HAS_RIO:
        transform = from_origin(0.0, 0.0, dx_km*1000.0, dx_km*1000.0)  # 假定左上角原点、像元 dx_km km
        profile = {
            "driver": "GTiff", "height": bathy.shape[0], "width": bathy.shape[1],
            "count": 1, "dtype": "float32", "transform": transform, "crs": None,
            "compress": "DEFLATE"
        }
        for arr, name in [(bathy,"terrain_bathy_m"),
                          (grav,"gravity_freeair_mgal"),
                          (mag,"magnetic_anomaly_nt")]:
            path = os.path.join(out_dir, name + ".tif")
            with rasterio.open(path, "w", **profile) as ds:
                ds.write(arr.astype("float32"), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=512, help="栅格高度")
    ap.add_argument("--W", type=int, default=512, help="栅格宽度")
    ap.add_argument("--dx_km", type=float, default=0.5, help="像元大小（km）")
    ap.add_argument("--seed", type=int, default=42, help="随机种子")
    args = ap.parse_args()

    H, W, dx_km, seed = args.H, args.W, args.dx_km, args.seed

    bathy = make_bathymetry(H, W, dx_km, seed=seed)
    grav  = make_gravity_freeair(bathy, dx_km, seed=seed+10)
    mag   = make_magnetic_anomaly(H, W, dx_km, seed=seed+20, stripe_angle_deg=30.0)

    out_dir = os.path.join(os.getcwd(), "synthetic_fields_out")
    save_outputs(bathy, grav, mag, out_dir, dx_km)

    print("Saved to:", out_dir)
    print("Files:")
    for fn in sorted(os.listdir(out_dir)):
        print(" -", fn)

if __name__ == "__main__":
    main()
