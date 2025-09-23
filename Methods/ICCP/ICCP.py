# ICCP implementation + demo
# This notebook implements the ICCP algorithm you described:
# - builds a 2D contour map (grid)
# - computes contour lines for many levels (matplotlib)
# - for each measurement point, finds a matching contour point:
#     * prefer nearest point on precomputed contour lines for the same level (or nearest level)
#     * fallback: use gradient-based projection (Newton-like) to move the point to the isocontour
# - computes optimal translation (mean of q - p') and iterates until convergence
#
# The demo generates a synthetic terrain with multiple Gaussian hills, simulates measurements
# taken along a path (with a true translation applied), adds noise, and recovers the translation.
#
# Notes:
# - Uses matplotlib to extract contour paths.
# - Uses scipy.spatial.cKDTree when available; otherwise falls back to numpy brute-force search.
# - Plots: contour map, measured points before/after correction, matched correspondences,
#   and translation norm per iteration.
#
# Run this cell to execute the demo and observe results.
# 不带旋转变换的ICCP算法

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree as KDTree
    KD_AVAILABLE = True
except Exception:
    KD_AVAILABLE = False


def make_contour_map(xmin=-50, xmax=50, ymin=-50, ymax=50, nx=301, ny=301, seed=0):
    rng = np.random.RandomState(seed)
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)
    centers = [(-20, -10), (10, -5), (15, 20), (-5, 25)]
    amps = [30, 20, 40, -25]
    sigmas = [8, 6, 10, 12]
    for c, a, s in zip(centers, amps, sigmas):
        Z += a * np.exp(-(((X-c[0])**2 + (Y-c[1])**2) / (2*s*s)))
    Z += 0.2 * X + 0.1 * Y + 3.0 * np.sin(0.15 * X) * np.cos(0.12 * Y)
    Z += 0.5 * rng.normal(size=Z.shape)
    return xs, ys, Z


def extract_contour_points(xs, ys, Z, levels_count=80):
    levels = np.linspace(np.min(Z), np.max(Z), levels_count)
    cs = plt.contour(xs, ys, Z, levels=levels)
    contour_points = {}
    for i, level in enumerate(cs.levels):
        pts = []
        coll = cs.collections[i]
        for path in coll.get_paths():
            v = path.vertices
            if v.size:
                pts.append(v)
        if pts:
            contour_points[level] = np.vstack(pts)
    plt.clf()
    return contour_points


def build_level_kdtrees(contour_points):
    kd = {}
    for level, pts in contour_points.items():
        if KD_AVAILABLE and pts.shape[0] > 0:
            kd[level] = KDTree(pts)
        else:
            kd[level] = pts
    return kd


def bilinear_interpolate(xs, ys, Z, xq, yq):
    xq = np.array(xq)
    yq = np.array(yq)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    ix = (xq - xs[0]) / dx
    iy = (yq - ys[0]) / dy
    ix0 = np.floor(ix).astype(int)
    iy0 = np.floor(iy).astype(int)
    ix0 = np.clip(ix0, 0, len(xs)-2)
    iy0 = np.clip(iy0, 0, len(ys)-2)
    tx = (ix - ix0)
    ty = (iy - iy0)
    z00 = Z[iy0, ix0]
    z10 = Z[iy0, ix0+1]
    z01 = Z[iy0+1, ix0]
    z11 = Z[iy0+1, ix0+1]
    z = (1-tx)*(1-ty)*z00 + tx*(1-ty)*z10 + (1-tx)*ty*z01 + tx*ty*z11
    return z


def interp_grad_from_precomputed(xs, ys, dz_dx, dz_dy, xq, yq):
    gx = bilinear_interpolate(xs, ys, dz_dx, xq, yq)
    gy = bilinear_interpolate(xs, ys, dz_dy, xq, yq)
    return gx, gy


def project_point_to_isocontour_gradient(xs, ys, Z, dz_dx, dz_dy, point, z_target, max_iter=20, tol=1e-3):
    q = np.array(point, dtype=float)
    for _ in range(max_iter):
        fval = bilinear_interpolate(xs, ys, Z, q[0], q[1])
        residual = z_target - fval
        if abs(residual) < tol:
            break
        gx, gy = interp_grad_from_precomputed(xs, ys, dz_dx, dz_dy, q[0], q[1])
        gnorm2 = gx*gx + gy*gy
        if gnorm2 < 1e-8:
            break
        step = np.array([gx, gy]) * (residual / gnorm2)
        q = q + step
    final_res = z_target - bilinear_interpolate(xs, ys, Z, q[0], q[1])
    return q, final_res


def find_nearest_contour_level(contour_points, desired_level):
    if len(contour_points)==0:
        return None
    levels = np.array(list(contour_points.keys()))
    idx = np.argmin(np.abs(levels - desired_level))
    return levels[idx]


def find_nearest_on_contour_kdtree_or_bruteforce(contour_points, kdtrees, level, point):
    if level not in contour_points:
        return None, np.inf
    pts = contour_points[level]
    if KD_AVAILABLE and isinstance(kdtrees.get(level, None), KDTree):
        d, idx = kdtrees[level].query(point)
        return pts[idx], d
    else:
        d2 = np.sum((pts - point)**2, axis=1)
        idx = np.argmin(d2)
        return pts[idx], np.sqrt(d2[idx])


def iccp(xs, ys, Z, P, T0=np.array([0.0,0.0]), eps=1e-3, max_iter=40, verbose=True):
    contour_points = extract_contour_points(xs, ys, Z, levels_count=100)
    kdtrees = build_level_kdtrees(contour_points)
    # Precompute gradients once
    dz_dy, dz_dx = np.gradient(Z, ys, xs)  # shape matches Z

    T = np.array(T0, dtype=float)
    history = [T.copy()]
    matched_history = []

    for k in range(max_iter):
        projected = P[:, :2] + T
        matched_q = np.zeros_like(projected)
        valid_mask = np.zeros(len(P), dtype=bool)
        for i, (pprime, zi) in enumerate(zip(projected, P[:,2])):
            if len(contour_points)==0:
                # fallback to gradient-only projection
                q, res = project_point_to_isocontour_gradient(xs, ys, Z, dz_dx, dz_dy, pprime, zi)
                matched_q[i] = q
                valid_mask[i] = True
                continue
            nearest_level = find_nearest_contour_level(contour_points, zi)
            q_candidate, d_candidate = find_nearest_on_contour_kdtree_or_bruteforce(contour_points, kdtrees, nearest_level, pprime)
            # Attempt gradient projection from pprime
            q_proj, res = project_point_to_isocontour_gradient(xs, ys, Z, dz_dx, dz_dy, pprime, zi, max_iter=12, tol=1e-3)
            if abs(res) < 5e-2:
                matched_q[i] = q_proj
                valid_mask[i] = True
            else:
                if q_candidate is not None:
                    matched_q[i] = q_candidate
                    valid_mask[i] = True
                else:
                    valid_mask[i] = False

        if np.sum(valid_mask) < 3:
            if verbose:
                print("Too few matches; stopping.")
            break

        q_valid = matched_q[valid_mask]
        pprime_valid = projected[valid_mask]
        deltaT = np.mean(q_valid - pprime_valid, axis=0)
        T = T + deltaT
        history.append(T.copy())
        matched_history.append((pprime_valid.copy(), q_valid.copy()))
        if verbose:
            print(f"Iter {k+1}: deltaT = {deltaT}, ||deltaT|| = {np.linalg.norm(deltaT):.6f}, matches = {len(q_valid)}")
        if np.linalg.norm(deltaT) < eps:
            if verbose:
                print("Converged.")
            break
    return T, np.array(history), matched_history


if __name__ == '__main__':
    # Demo with fewer points for speed
    xs, ys, Z = make_contour_map(nx=301, ny=301, seed=1)

    # generate path and samples
    t = np.linspace(-28, 28, 40)  # fewer points
    true_path = np.vstack((t, 7.0 * np.sin(0.12 * t))).T
    T_true = np.array([7.3, -4.7])                  # 预设的偏移量
    z_samples = bilinear_interpolate(xs, ys, Z, true_path[:, 0] + T_true[0], true_path[:, 1] + T_true[1])
    z_noisy = z_samples + 0.4 * np.random.randn(len(z_samples))
    pos_noisy = true_path + 0.3 * np.random.randn(*true_path.shape)
    P = np.column_stack((pos_noisy, z_noisy))       # 实际获取的数据，位置是有偏的，测得的地形数据存在一定噪声

    T0 = np.array([-10.0, 3.0])
    T_est, history, matched = iccp(xs, ys, Z, P, T0=T0, eps=1e-3, max_iter=40, verbose=True)

    print("\nTrue translation:", T_true)
    print("Initial guess:", T0)
    print("Estimated translation:", T_est)
    print("Error (est - true):", T_est - T_true)

    # Plot results
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    # levels = np.linspace(np.min(Z), np.max(Z), 40)
    # ax.contour(xs, ys, Z, levels=levels)
    proj_initial = P[:, :2] + T0
    proj_est = P[:, :2] + T_est
    proj_true = np.vstack((true_path[:, 0] + T_true[0], true_path[:, 1] + T_true[1])).T
    ax.scatter(proj_initial[:, 0], proj_initial[:, 1], marker='x', label='First guess')
    ax.scatter(proj_est[:, 0], proj_est[:, 1], marker='o', label='Final guess')
    ax.scatter(proj_true[:, 0], proj_true[:, 1], marker='^', label='True line')
    if len(matched) > 0:
        pprime_valid, q_valid = matched[-1]
        for pp, qq in zip(pprime_valid[::2], q_valid[::2]):
            ax.plot([pp[0], qq[0]], [pp[1], qq[1]])
    ax.set_title("ICCP demo: contours + projected points (initial and corrected)")
    ax.set_xlabel("X");
    ax.set_ylabel("Y")
    plt.legend()
    plt.show()

    fig2 = plt.figure(figsize=(6, 4))
    ax2 = fig2.add_subplot(1, 1, 1)
    deltas = np.linalg.norm(np.diff(history, axis=0, prepend=history[0:1]), axis=1)
    ax2.plot(np.arange(len(history)), deltas)
    ax2.set_xlabel("Iteration");
    ax2.set_ylabel("||deltaT||");
    ax2.set_title("Translation step norms per iter")
    plt.show()

    print("Translation history:")
    for i, h in enumerate(history):
        print(f" Iter {i}: T = [{h[0]:.4f}, {h[1]:.4f}]")
