# ICCP with 2D rigid transform (rotation + translation) implementation + demo
# This notebook cell implements the ICCP algorithm extended to 2D rigid transforms:
#   p_transformed = R(theta) * p + T
# It finds correspondences (nearest contour points or gradient-projected isocontour points)
# and at each iteration computes the optimal 2D rigid transform (rotation + translation)
# between the transformed measurement points and their matched contour points using SVD (Procrustes / Umeyama style).
#
# The transform update composes the new incremental transform with the current one.
# A demo with synthetic terrain, a ground-truth rigid transform (rotation+translation),
# and noisy samples is included. The script plots contour map, initial & corrected positions,
# and reports estimation errors.
#
# Requires: numpy, matplotlib, scipy (optional but improves performance for KDTree).
# 标准版ICCP

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.spatial import cKDTree as KDTree
    KD_AVAILABLE = True
except Exception:
    KD_AVAILABLE = False

# ------------------- utilities (grid, contours, interpolation) -------------------
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

# precomputed gradients interpolation
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

# ------------------- rigid transform (2D) estimator -------------------
def compute_rigid_transform_2d(A, B):
    """
    Compute rotation R (2x2) and translation t (2,) that best align A -> B
    minimizing sum ||R A_i + t - B_i||^2 (no scaling).
    A and B are Nx2 arrays. Returns R, t.
    """
    assert A.shape == B.shape and A.shape[1] == 2
    n = A.shape[0]
    muA = A.mean(axis=0)
    muB = B.mean(axis=0)
    A_c = A - muA
    B_c = B - muB
    H = A_c.T @ B_c  # 2x2
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # ensure right-handed (det=+1)
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    t = muB - R @ muA
    return R, t

def rotation_matrix_from_angle(theta):
    c = np.cos(theta); s = np.sin(theta)
    return np.array([[c, -s],[s, c]])

def angle_from_rotation_matrix(R):
    # angle such that R = [[cos,-sin],[sin,cos]]
    return np.arctan2(R[1,0], R[0,0])

# ------------------- ICCP loop for rigid 2D transform -------------------
def iccp_rigid(xs, ys, Z, P, R0=None, T0=None, eps=1e-4, ang_eps=1e-4, max_iter=50, verbose=True, levels_count=100):
    """
    ICCP for 2D rigid transform.
    P: Nx3 array of measurement points [x,y,z_measure]
    R0: initial rotation matrix (2x2) or None
    T0: initial translation (2,) or None
    Returns: R_est, T_est, history (list of (R,T)), matched_history
    """
    contour_points = extract_contour_points(xs, ys, Z, levels_count=levels_count)
    kdtrees = build_level_kdtrees(contour_points)
    dz_dy, dz_dx = np.gradient(Z, ys, xs)
    n = len(P)
    if R0 is None:
        R = np.eye(2)
    else:
        R = np.array(R0, dtype=float)
    if T0 is None:
        T = np.zeros(2, dtype=float)
    else:
        T = np.array(T0, dtype=float)
    history = [(R.copy(), T.copy())]
    matched_history = []
    for k in range(max_iter):
        # transform points with current estimate
        src = (R @ P[:, :2].T).T + T  # Nx2
        matched_q = np.zeros_like(src)
        valid_mask = np.zeros(n, dtype=bool)
        dists = np.full(n, np.inf)
        for i, (ptrans, zi) in enumerate(zip(src, P[:,2])):
            if len(contour_points)==0:
                q, res = project_point_to_isocontour_gradient(xs, ys, Z, dz_dx, dz_dy, ptrans, zi)
                matched_q[i] = q
                valid_mask[i] = True
                dists[i] = np.linalg.norm(q - ptrans)
                continue
            nearest_level = find_nearest_contour_level(contour_points, zi)
            q_cand, d_cand = find_nearest_on_contour_kdtree_or_bruteforce(contour_points, kdtrees, nearest_level, ptrans)
            # attempt gradient projection from ptrans
            q_proj, res = project_point_to_isocontour_gradient(xs, ys, Z, dz_dx, dz_dy, ptrans, zi, max_iter=12, tol=1e-3)
            if abs(res) < 5e-2:
                matched_q[i] = q_proj
                dists[i] = np.linalg.norm(q_proj - ptrans)
                valid_mask[i] = True
            else:
                if q_cand is not None:
                    matched_q[i] = q_cand
                    dists[i] = d_cand
                    valid_mask[i] = True
                else:
                    valid_mask[i] = False
                    dists[i] = np.inf
        if np.sum(valid_mask) < 3:
            if verbose:
                print("Too few matches; stopping. Valid:", np.sum(valid_mask))
            break
        A = src[valid_mask]   # source points (transformed)
        B = matched_q[valid_mask]  # target correspondence on contour
        # compute incremental rigid transform R_delta, t_delta that maps A -> B
        R_delta, t_delta = compute_rigid_transform_2d(A, B)
        # update global transform: new_R = R_delta * R; new_T = R_delta * T + t_delta
        R = R_delta @ R
        T = R_delta @ T + t_delta
        history.append((R.copy(), T.copy()))
        matched_history.append((A.copy(), B.copy(), dists[valid_mask].copy()))
        # check convergence: translation norm and rotation change
        trans_norm = np.linalg.norm(t_delta)
        ang = np.abs(angle_from_rotation_matrix(R_delta))
        if verbose:
            print(f"Iter {k+1}: trans_delta = {t_delta}, ||t||={trans_norm:.6f}, ang_delta={ang:.6f} rad, matches={len(A)}")
        if trans_norm < eps and ang < ang_eps:
            if verbose:
                print("Converged.")
            break
    return R, T, history, matched_history

# ------------------- Demo -------------------
np.random.seed(123)
xs, ys, Z = make_contour_map(nx=301, ny=301, seed=2)
# measurement path (in sensor frame before transform)
t = np.linspace(-28, 28, 50)
path = np.vstack((t, 6.2*np.sin(0.12*t))).T  # Nx2
# true rigid transform (rotation + translation)
theta_true = np.deg2rad(18.5)  # rotation of 18.5 degrees
R_true = rotation_matrix_from_angle(theta_true)
T_true = np.array([6.7, -3.9])
# transformed positions in map frame
map_positions = (R_true @ path.T).T + T_true
# sample z from grid at transformed positions
z_samples = bilinear_interpolate(xs, ys, Z, map_positions[:,0], map_positions[:,1])
# add noise to z and small pos noise
z_noisy = z_samples + 0.35 * np.random.randn(len(z_samples))
pos_noisy = path + 0.25 * np.random.randn(*path.shape)
P = np.column_stack((pos_noisy, z_noisy))

# initial guess - deliberately wrong rotation & translation
theta0 = np.deg2rad(-10.0)
R0 = rotation_matrix_from_angle(theta0)
T0 = np.array([-8.0, 5.0])

# run ICCP rigid
R_est, T_est, history, matched = iccp_rigid(xs, ys, Z, P, R0=R0, T0=T0, eps=1e-3, ang_eps=1e-4, max_iter=60, verbose=True, levels_count=100)

theta_est = angle_from_rotation_matrix(R_est)
print("\nTrue theta (deg), T:", np.rad2deg(theta_true), T_true)
print("Estimated theta (deg), T:", np.rad2deg(theta_est), T_est)
print("Errors: theta_err(deg) =", np.rad2deg(theta_est - theta_true), " T_err =", T_est - T_true)

# visualize
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
levels = np.linspace(np.min(Z), np.max(Z), 40)
ax.contour(xs, ys, Z, levels=levels, linewidths=0.7)
# initial projected points (with initial guess)
proj_initial = (R0 @ P[:,:2].T).T + T0            # 初始值
proj_est = (R_est @ P[:,:2].T).T + T_est          # 最终预测值
proj_true = (R_true @ P[:, :2].T).T + T_true      # 真实值
ax.scatter(proj_initial[:,0], proj_initial[:,1], marker='x', label='proj_initial')
ax.scatter(proj_est[:,0], proj_est[:,1], marker='o', label='proj_est')
ax.scatter(proj_true[:, 0], proj_true[:, 1], marker='^', label='True line')

# draw matched correspondences from last iter
if len(matched)>0:
    A_last, B_last, d_last = matched[-1]
    for a,b in zip(A_last[::2], B_last[::2]):
        ax.plot([a[0], b[0]], [a[1], b[1]], alpha=0.6)
ax.set_title("ICCP (2D rigid) demo: contours + projected points (initial vs corrected)")
ax.set_xlabel("X"); ax.set_ylabel("Y")
ax.legend()
plt.show()

# convergence summary
trans_steps = [np.linalg.norm(t) for (_,t) in history]
ang_steps = [np.abs(angle_from_rotation_matrix(R)) for (R,_) in history]
fig2 = plt.figure(figsize=(6,4))
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(np.arange(len(history)), trans_steps, label='||T||')
ax2.set_xlabel("Iteration"); ax2.set_ylabel("||T||"); ax2.set_title("Translation magnitude per iter")
ax2.grid(True)
plt.show()

print("\nFinal transform history (last few):")
for i,(R_h,T_h) in enumerate(history[-10:], start=len(history)-10):
    print(f" Iter {i}: theta(deg)={np.rad2deg(angle_from_rotation_matrix(R_h)):.4f}, T=[{T_h[0]:.4f}, {T_h[1]:.4f}]")

# End of demo cell.
