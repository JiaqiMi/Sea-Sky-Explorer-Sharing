# Continuous-time ICCP prototype (B-spline parameterization for SE(2) trajectory)
# - Map residuals: f(x,y) - z
# - IMU yaw prior: differences of theta(t) constrained by integrated yaw from IMU
# - Cubic B-spline parameterization using scipy.interpolate.BSpline
# - Analytic Jacobian assembled (dense) for least_squares
#
# Demo: synthetic map, synthetic true spline trajectory, noisy sensor samples and IMU yaw.
# Parameters are exposed at the top for easy tuning.
# ICCP的改进版本，引入IMU数据
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

# ---------------- Parameters (tune these) ----------------
np.random.seed(42)
GRID_XY = (-50,50,-50,50)    # map extent (xmin,xmax,ymin,ymax)
NX, NY = 301, 301            # map resolution
BSPLINE_DEGREE = 3          # cubic
N_CONTROL = 18              # number of control points per channel (px,py,theta)
MEAS_N = 120                # number of map-sampled measurements
IMU_RATE = 50.0             # Hz for synthetic IMU (used to integrate yaw)
SIG_Z = 0.4                 # measurement noise in z (map value)
SIG_POS_NOISE = 0.15        # sensor position noise in local frame
SIG_IMU_YAW_RATE = 0.01    # IMU yaw rate noise std (rad/s)
MAP_NOISE = 0.3             # small map noise
W_IMU = 1.0                 # IMU residual weight (applied as scaling)
W_SMOOTH = 0.1              # smoothness weight (finite-diff on control points)
# --------------------------------------------------------

# ---------------- Helper: map creation and interpolation ----------------
xmin,xmax,ymin,ymax = GRID_XY
xs = np.linspace(xmin, xmax, NX)
ys = np.linspace(ymin, ymax, NY)
Xg, Yg = np.meshgrid(xs, ys)
# synthetic terrain: sum of gaussians + ripples
Zg = (30*np.exp(-(((Xg+20)**2+(Yg+10)**2)/(2*8*8)))
      + 20*np.exp(-(((Xg-10)**2+(Yg+5)**2)/(2*6*6)))
      + 40*np.exp(-(((Xg-15)**2+(Yg-20)**2)/(2*10*10)))
      - 25*np.exp(-(((Xg+5)**2+(Yg-25)**2)/(2*12*12)))
      + 0.2*Xg + 0.1*Yg + 3.0*np.sin(0.15*Xg)*np.cos(0.12*Yg))
Zg += MAP_NOISE * np.random.randn(*Zg.shape)

# precompute gradients on grid (for nabla f)
dz_dy, dz_dx = np.gradient(Zg, ys, xs)  # note order: gradient returns grads for each axis

def bilinear_interpolate_grid(xs, ys, Z, xq, yq):
    # supports vectors
    xq = np.asarray(xq); yq = np.asarray(yq)
    dx = xs[1] - xs[0]; dy = ys[1] - ys[0]
    ix = (xq - xs[0]) / dx
    iy = (yq - ys[0]) / dy
    ix0 = np.floor(ix).astype(int)
    iy0 = np.floor(iy).astype(int)
    ix0 = np.clip(ix0, 0, len(xs)-2)
    iy0 = np.clip(iy0, 0, len(ys)-2)
    tx = (ix - ix0); ty = (iy - iy0)
    z00 = Z[iy0, ix0]; z10 = Z[iy0, ix0+1]; z01 = Z[iy0+1, ix0]; z11 = Z[iy0+1, ix0+1]
    z = (1-tx)*(1-ty)*z00 + tx*(1-ty)*z10 + (1-tx)*ty*z01 + tx*ty*z11
    return z

def interp_grad_at(xs, ys, dZdx, dZdy, xq, yq):
    gx = bilinear_interpolate_grid(xs, ys, dZdx, xq, yq)
    gy = bilinear_interpolate_grid(xs, ys, dZdy, xq, yq)
    return gx, gy

# ---------------- Synthetic true trajectory (B-spline control points) ----------------
# time span
t0, tf = 0.0, 60.0  # seconds
times_meas = np.linspace(t0, tf, MEAS_N)
# define control times for spline (uniform)
m = N_CONTROL
k = BSPLINE_DEGREE
# knot vector: clamped uniform
# number of knots = ncoef + k + 1
ncoef = m
# inner knots (uniform) from t0 to tf
inner_knots = np.linspace(t0, tf, ncoef - k + 1)  # ensures correct count
# build full knots with clamping at ends
knots = np.concatenate(([t0]*(k), inner_knots, [tf]*(k)))
# check length: len(knots) == ncoef + k + 1
assert len(knots) == ncoef + k + 1

# create ground truth control points for px, py, theta
# a smooth L-shaped-ish path
cpx_true = np.linspace(-30, 30, ncoef) + 4.0*np.sin(np.linspace(0,3*np.pi,ncoef))
cpy_true = 6.0*np.sin(np.linspace(-1.5,1.5,ncoef)) * np.linspace(1,1,ncoef) + np.linspace(-5,5,ncoef)
# theta true: moderate turning
ctheta_true = np.deg2rad(10.0*np.sin(np.linspace(0,2*np.pi,ncoef)) + np.linspace(0,20,ncoef)/2.0)

# build BSpline objects for evaluation (separately for each channel)
spline_px = BSpline(knots, cpx_true, k)
spline_py = BSpline(knots, cpy_true, k)
spline_theta = BSpline(knots, ctheta_true, k)

def eval_spline_all(tq, cp_x, cp_y, cp_theta):
    # create BSpline with given coeffs and evaluate
    sx = BSpline(knots, cp_x, k)
    sy = BSpline(knots, cp_y, k)
    st = BSpline(knots, cp_theta, k)
    return np.column_stack((sx(tq), sy(tq), st(tq)))

# sample map-measurements: define sensor-frame positions along a trajectory (e.g., vehicle offset points)
# choose some body-frame measured points (e.g., location of sensor relative to body origin). Here p_sensor=0 (sensor at origin)
p_sensor = np.array([0.0, 0.0])  # if sensor gives local position; for simplicity, use body origin positions
# generate true positions in map frame at measurement times
poses_true = eval_spline_all(times_meas, cpx_true, cpy_true, ctheta_true)  # Nx3 px,py,theta
positions_true = poses_true[:, :2]
# for generality, let's assume measurements are at body-frame points p_sensor; so map point = R(theta)*p_sensor + p(t) => p(t)
map_points = positions_true  # since p_sensor is zero
# sample z value from grid at map_points
z_samples = bilinear_interpolate_grid(xs, ys, Zg, map_points[:,0], map_points[:,1])
# add measurement noise to z and small positional noise in sensor local frame
z_noisy = z_samples + SIG_Z * np.random.randn(len(z_samples))
pos_noisy_body = (np.column_stack((np.linspace(-0.5,0.5,len(times_meas)), 0.1*np.sin(np.linspace(0,3,len(times_meas)))))
                  + SIG_POS_NOISE * np.random.randn(len(times_meas),2))
# sensor measures p_sensor in body coordinates; but here body path is positions_true, so sensor in map frame is R*pos_noisy_body+p(t)
# We'll simulate body-frame measured offsets and then transform using true pose to get p_meas_map for residual mapping.
Rts = [np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]]) for th in poses_true[:,2]]
map_points_noisy = np.array([Rts[i] @ pos_noisy_body[i] + positions_true[i] for i in range(len(times_meas))])
# measure z at noisy map points (simulate small positioning errors before sampling)
z_sample_noisy_positions = bilinear_interpolate_grid(xs, ys, Zg, map_points_noisy[:,0], map_points_noisy[:,1])
# We'll take observed z as these values plus measurement noise
z_obs = z_sample_noisy_positions + SIG_Z * np.random.randn(len(z_sample_noisy_positions))

# ---------------- Synthetic IMU yaw-rate data (for yaw prior) ----------------
# simulate high-rate IMU yaw rates by differentiating theta_true and adding noise
imu_dt = 1.0 / IMU_RATE
imu_times = np.arange(t0, tf+1e-6, imu_dt)
theta_true_fine = spline_theta(imu_times)
yaw_rates_true = np.gradient(theta_true_fine, imu_dt)  # approximate derivative
# add noise to yaw rates
yaw_rates_meas = yaw_rates_true + SIG_IMU_YAW_RATE * np.random.randn(len(yaw_rates_true))
# form integrated yaw increments between measurement times times_meas for priors
# for each pair (i -> i+1) we'll compute integrated yaw from imu_times
def integrate_imu_yaw(imu_t, yaw_rates, ta, tb):
    # simple trapezoidal integration between ta and tb
    mask = (imu_t>=ta) & (imu_t<=tb)
    if np.sum(mask) < 2:
        # fallback: linear interpolation
        return (np.interp(tb, imu_t, np.cumsum(yaw_rates*imu_dt)) - np.interp(ta, imu_t, np.cumsum(yaw_rates*imu_dt)))
    tsel = imu_t[mask]; yr = yaw_rates[mask]
    # trapezoid
    return np.trapz(yr, tsel)

# compute integrated yaw increments for adjacent measurement times (as priors)
imu_dyaw = []
for i in range(len(times_meas)-1):
    ta = times_meas[i]; tb = times_meas[i+1]
    dy = integrate_imu_yaw(imu_times, yaw_rates_meas, ta, tb)
    imu_dyaw.append(dy)
imu_dyaw = np.array(imu_dyaw)  # size MEAS_N-1

# ---------------- B-spline basis evaluation matrix (for all control indices) ----------------
# We need basis function values N_j(t_i) for each control coefficient j at each measurement time.
# We'll compute basis matrix B: shape (MEAS_N, ncoef) where row i has basis values for t_i across coefficients.
def compute_basis_matrix(tq):
    # for each coefficient index, create unit coefficient vector and evaluate BSpline at tq
    B = np.zeros((len(tq), ncoef))
    # we reuse BSpline with coeffs = e_j
    for j in range(ncoef):
        coeffs = np.zeros(ncoef); coeffs[j]=1.0
        S = BSpline(knots, coeffs, k)
        B[:, j] = S(tq)
    return B  # (Ntimes, ncoef)

B_meas = compute_basis_matrix(times_meas)  # shape (MEAS_N, ncoef)
B_imu_pairs = []  # for each adjacent pair (i,i+1), basis difference for theta control points at tb - ta
for i in range(len(times_meas)-1):
    Bt = compute_basis_matrix([times_meas[i]])[0]
    Bt1 = compute_basis_matrix([times_meas[i+1]])[0]
    B_imu_pairs.append(Bt1 - Bt)
B_imu_pairs = np.array(B_imu_pairs)  # shape (MEAS_N-1, ncoef)

# ---------------- Build residual and analytic Jacobian for least_squares ----------------
# parameter vector x = [cpx (ncoef), cpy (ncoef), ctheta (ncoef)] length 3*ncoef
def pack_params(cpx, cpy, cth):
    return np.concatenate([cpx, cpy, cth])

def unpack_params(x):
    cpx = x[0:ncoef]
    cpy = x[ncoef:2*ncoef]
    cth = x[2*ncoef:3*ncoef]
    return cpx, cpy, cth

def residuals_and_jac(x):
    cpx, cpy, cth = unpack_params(x)
    # evaluate poses at measurement times quickly using basis matrix
    px_t = B_meas @ cpx   # (MEAS_N,)
    py_t = B_meas @ cpy
    th_t = B_meas @ cth
    # rotation matrices (vectorized)
    cos_t = np.cos(th_t); sin_t = np.sin(th_t)
    # transformed map sample point from sensor body offset pos_noisy_body
    # For our simplified model, sensors measure body-frame offsets pos_noisy_body which we map: x_i = R(th_i)*pos_noisy_body[i] + [px,py]
    pts_map_x = cos_t * pos_noisy_body[:,0] - sin_t * pos_noisy_body[:,1] + px_t
    pts_map_y = sin_t * pos_noisy_body[:,0] + cos_t * pos_noisy_body[:,1] + py_t
    # map residuals r_map = f(x_i) - z_obs
    fvals = bilinear_interpolate_grid(xs, ys, Zg, pts_map_x, pts_map_y)
    r_map = fvals - z_obs  # shape (MEAS_N,)
    # IMU residuals: for each adjacent pair i->i+1: (theta(t_{i+1}) - theta(t_i)) - imu_dyaw[i]
    th_diff = th_t[1:] - th_t[:-1]
    r_imu = (th_diff - imu_dyaw) * np.sqrt(W_IMU)  # scale by weight (simple scaling)
    # smoothness residual: finite-diff second-order on control points for translation and theta
    # for each j compute c_{j+2} - 2 c_{j+1} + c_j for j=0..ncoef-3
    if W_SMOOTH > 0:
        r_smooth_px = (cpx[2:] - 2*cpx[1:-1] + cpx[:-2]) * np.sqrt(W_SMOOTH)
        r_smooth_py = (cpy[2:] - 2*cpy[1:-1] + cpy[:-2]) * np.sqrt(W_SMOOTH)
        r_smooth_th = (cth[2:] - 2*cth[1:-1] + cth[:-2]) * np.sqrt(W_SMOOTH)
        r_smooth = np.concatenate([r_smooth_px, r_smooth_py, r_smooth_th])
    else:
        r_smooth = np.zeros(0)
    # stack residuals
    r = np.concatenate([r_map, r_imu, r_smooth])
    # Jacobian assembly: shape (len(r), 3*ncoef)
    m_r = len(r_map); m_imu = len(r_imu); m_smooth = len(r_smooth)
    J = np.zeros((m_r + m_imu + m_smooth, 3*ncoef))
    # Map residual jacobians:
    # r_i = f(x_i) - z  -> dr/dx = nabla f(x_i) (1x2) ; dx/dcpx = d(px)/dcpx + d(Rp)/dcpx (but pos_noisy_body depends zero on cpx)
    # Since pos_noisy_body is measured in body, derivative contributions separate:
    # derivative wrt cpx_j: partial x_i / partial cpx_j = B_meas[i,j] ; partial y_i / partial cpx_j = 0 -> combine with grad
    # likewise for cpy_j
    # derivative wrt cth_j: partial x_i / partial th = (dR/dth * p_body)_x * B_meas[i,j], similar for y
    # compute nabla f at pts_map
    grad_fx, grad_fy = interp_grad_at(xs, ys, dz_dx, dz_dy, pts_map_x, pts_map_y)  # arrays length MEAS_N
    # fill jac for cpx block (columns 0:ncoef)
    for j in range(ncoef):
        Nj = B_meas[:, j]  # (MEAS_N,)
        # dr_map/dcpx_j = grad_fx * Nj  (since dx/dcpx_j = Nj)
        J[0:m_r, j] = grad_fx * Nj
        # dr_map/dcpy_j = grad_fy * Nj (since dy/dcpy_j = Nj)
        J[0:m_r, ncoef + j] = grad_fy * Nj
        # dr_map/dcth_j = grad_fx * (d x / dtheta_j) + grad_fy * (d y / dtheta_j)
        # where d x / dtheta_j = (dR/dtheta * p_body)_x * Nj, similarly for y
        # compute dR/dtheta * p for each measurement: use vectorized form per measurement
        # dR/dtheta = [[-sin, -cos],[cos, -sin]]
        dRx = (-sin_t * pos_noisy_body[:,0] - cos_t * pos_noisy_body[:,1])  # (MEAS_N,)
        dRy = ( cos_t * pos_noisy_body[:,0] - sin_t * pos_noisy_body[:,1])
        J[0:m_r, 2*ncoef + j] = (grad_fx * dRx + grad_fy * dRy) * Nj
    # IMU jacobians: r_imu_i = (theta_{i+1} - theta_i - imu_dyaw_i)*sqrt(W_IMU)
    # theta at times = B_meas @ cth, so derivative wrt cth_j is (B_{i+1,j} - B_{i,j}) * sqrt(W_IMU)
    for i in range(m_imu):
        J[m_r + i, 2*ncoef : 3*ncoef] = B_imu_pairs[i] * np.sqrt(W_IMU)
    # Smooth jacobians: each second-diff residual corresponds to three control indices j,j+1,j+2
    if m_smooth > 0:
        row0 = m_r + m_imu
        # px smooth block
        for idx in range(ncoef-2):
            row = row0 + idx
            J[row, idx] = 1.0 * np.sqrt(W_SMOOTH)
            J[row, idx+1] = -2.0 * np.sqrt(W_SMOOTH)
            J[row, idx+2] = 1.0 * np.sqrt(W_SMOOTH)
        # py smooth block
        row0_py = row0 + (ncoef-2)
        for idx in range(ncoef-2):
            row = row0_py + idx
            J[row, ncoef + idx] = 1.0 * np.sqrt(W_SMOOTH)
            J[row, ncoef + idx+1] = -2.0 * np.sqrt(W_SMOOTH)
            J[row, ncoef + idx+2] = 1.0 * np.sqrt(W_SMOOTH)
        # th smooth block
        row0_th = row0_py + (ncoef-2)
        for idx in range(ncoef-2):
            row = row0_th + idx
            J[row, 2*ncoef + idx] = 1.0 * np.sqrt(W_SMOOTH)
            J[row, 2*ncoef + idx+1] = -2.0 * np.sqrt(W_SMOOTH)
            J[row, 2*ncoef + idx+2] = 1.0 * np.sqrt(W_SMOOTH)
    return r, J

# ---------------- Initial guess for control points (use perturbed true + INS-like noise) ----------------
# create initial control points by sampling true spline and adding noise
cpx_init = cpx_true + 0.8 * np.random.randn(ncoef)
cpy_init = cpy_true + 0.8 * np.random.randn(ncoef)
cth_init = ctheta_true + np.deg2rad(2.0) * np.random.randn(ncoef)  # small yaw bias noise

x0 = pack_params(cpx_init, cpy_init, cth_init)

# ---------------- Run least_squares (LM) with analytic jacobian ----------------
print("Starting optimization...")
res = least_squares(lambda x: residuals_and_jac(x)[0], x0, jac=lambda x: residuals_and_jac(x)[1], method='lm', xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=200)
print("Optimization done. success:", res.success, "message:", res.message)
x_opt = res.x
cpx_opt, cpy_opt, cth_opt = unpack_params(x_opt)

# ---------------- Evaluate results ----------------
poses_init = eval_spline_all(times_meas, cpx_init, cpy_init, cth_init)
poses_opt = eval_spline_all(times_meas, cpx_opt, cpy_opt, cth_opt)
poses_true = poses_true  # from before
# compute position RMSE and yaw RMSE on measurement times
pos_err_init = np.linalg.norm(poses_init[:,0:2] - poses_true[:,0:2], axis=1)
pos_err_opt = np.linalg.norm(poses_opt[:,0:2] - poses_true[:,0:2], axis=1)
yaw_err_init = np.abs(np.unwrap(poses_init[:,2]) - np.unwrap(poses_true[:,2]))
yaw_err_opt = np.abs(np.unwrap(poses_opt[:,2]) - np.unwrap(poses_true[:,2]))

print(f"Position RMSE initial: {np.sqrt(np.mean(pos_err_init**2)):.3f}  optimized: {np.sqrt(np.mean(pos_err_opt**2)):.3f}")
print(f"Yaw RMSE initial(deg): {np.rad2deg(np.sqrt(np.mean(yaw_err_init**2))):.3f}  optimized: {np.rad2deg(np.sqrt(np.mean(yaw_err_opt**2))):.3f}")

# ---------------- Visualizations ----------------
fig, ax = plt.subplots(1,2,figsize=(15,6))
# map + trajectories
ax[0].contour(xs, ys, Zg, levels=30, linewidths=0.6)
ax[0].plot(poses_true[:,0], poses_true[:,1], 'k-', label='true path')
ax[0].plot(poses_init[:,0], poses_init[:,1], 'r--', label='init path')
ax[0].plot(poses_opt[:,0], poses_opt[:,1], 'g-', linewidth=2, label='opt path')
ax[0].scatter(map_points_noisy[:,0], map_points_noisy[:,1], c='orange', s=10, label='z sample locations (noisy)')
ax[0].legend(); ax[0].set_title("Map + trajectories (true / init / optimized)"); ax[0].set_xlabel("X"); ax[0].set_ylabel("Y")

# errors
ax[1].plot(times_meas, pos_err_init, label='pos err init')
ax[1].plot(times_meas, pos_err_opt, label='pos err opt')
ax[1].plot(times_meas, np.rad2deg(yaw_err_init), '--', label='yaw err init (deg)')
ax[1].plot(times_meas, np.rad2deg(yaw_err_opt), '-', label='yaw err opt (deg)')
ax[1].legend(); ax[1].set_title("Per-sample errors"); ax[1].set_xlabel("time (s)"); ax[1].grid(True)

plt.show()

# plot residuals distribution before/after for map residuals
r_init, _ = residuals_and_jac(x0)
r_opt, _ = residuals_and_jac(x_opt)
m_r = MEAS_N
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.hist(r_init[:m_r], bins=30); plt.title("Map residuals (init)")
plt.subplot(1,2,2)
plt.hist(r_opt[:m_r], bins=30); plt.title("Map residuals (opt)")
plt.show()

# show control point comparison for theta
plt.figure(figsize=(8,4))
plt.plot(np.arange(ncoef), ctheta_true, label='true theta ctrl pts')
plt.plot(np.arange(ncoef), cth_init, 'r--', label='init theta ctrl pts')
plt.plot(np.arange(ncoef), cth_opt, 'g-', label='opt theta ctrl pts')
plt.legend(); plt.title("Theta control points (true / init / opt)")
plt.show()

# Provide an interface summary for parameter tuning
print("\n--- Parameter tuning interface (current values) ---")
print(f"N_CONTROL = {N_CONTROL}, BSPLINE_DEGREE = {BSPLINE_DEGREE}, MEAS_N = {MEAS_N}")
print(f"SIG_Z = {SIG_Z}, SIG_POS_NOISE = {SIG_POS_NOISE}, SIG_IMU_YAW_RATE = {SIG_IMU_YAW_RATE}")
print(f"W_IMU = {W_IMU}, W_SMOOTH = {W_SMOOTH}")
print("To adjust performance, change these variables at the top and re-run the script.")
