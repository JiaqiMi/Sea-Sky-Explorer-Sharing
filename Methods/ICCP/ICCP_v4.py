# Continuous-time ICCP prototype with higher-order B-spline (quintic) and IMU pre-integration
# - Parameterization: cubic -> replaced by quintic (degree = 5)
# - IMU pre-integration: integrate angular rates and body-frame accelerations to produce
#   delta_theta and delta_p_body between subsequent measurement timestamps.
# - Residuals: map residuals (f(x)-z), IMU rotation residuals (delta theta), IMU translation residuals (R_a^T (p_b-p_a) - delta_p_body)
# - Analytic Jacobians are assembled (dense) and used with scipy least_squares (LM).
#
# Requirements: numpy, scipy, matplotlib
# Save as a script or run in a notebook. This demo creates synthetic consistent IMU data from the ground truth spline.
# ICCP的改进版本，增加了IMU，使用更高阶的 spline（已为 cubic）并将 IMU 预积分替换进来
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.optimize import least_squares

# ---------------- Parameters (tune these) ----------------
np.random.seed(1)
GRID_XY = (-50,50,-50,50)    # map extent (xmin,xmax,ymin,ymax)
NX, NY = 301, 301            # map resolution
BSPLINE_DEGREE = 5          # quintic
N_CONTROL = 22              # number of control points per channel (px,py,theta)
MEAS_N = 140                # number of map-sampled measurements
IMU_RATE = 100.0             # Hz for synthetic IMU (used to integrate yaw / accel)
SIG_Z = 0.35                 # measurement noise in z (map value)
SIG_POS_NOISE = 0.10         # sensor position noise in local frame
SIG_IMU_GYRO = 0.005         # gyro noise std (rad/s)
SIG_IMU_ACC = 0.05           # accel noise std (m/s^2) in body frame
MAP_NOISE = 0.2              # small map noise
W_IMU_ROT = 1.0              # weight for IMU rotation residual (scaling)
W_IMU_TRANS = 0.5            # weight for IMU translation residual (scaling)
W_SMOOTH = 0.05              # smoothness weight (finite-diff on control points)
# -----------------------------------------------------------------

# ---------------- Helper: map creation and interpolation ----------------
xmin,xmax,ymin,ymax = GRID_XY
xs = np.linspace(xmin, xmax, NX)
ys = np.linspace(ymin, ymax, NY)
Xg, Yg = np.meshgrid(xs, ys)
# synthetic terrain
Zg = (30*np.exp(-(((Xg+20)**2+(Yg+10)**2)/(2*8*8)))
      + 20*np.exp(-(((Xg-10)**2+(Yg+5)**2)/(2*6*6)))
      + 40*np.exp(-(((Xg-15)**2+(Yg-20)**2)/(2*10*10)))
      - 25*np.exp(-(((Xg+5)**2+(Yg-25)**2)/(2*12*12)))
      + 0.15*Xg + 0.08*Yg + 2.5*np.sin(0.13*Xg)*np.cos(0.11*Yg))
Zg += MAP_NOISE * np.random.randn(*Zg.shape)

# grid gradients
dz_dy, dz_dx = np.gradient(Zg, ys, xs)

def bilinear_interpolate_grid(xs, ys, Z, xq, yq):
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

# ---------------- B-spline knots (clamped) helper ----------------
def make_clamped_knots(t0, tf, ncoef, k):
    # number of inner knots = ncoef - k - 1 (can be zero or more)
    inner_count = ncoef - k - 1
    if inner_count > 0:
        inner = np.linspace(t0, tf, inner_count + 2)[1:-1]
    else:
        inner = np.array([])
    knots = np.concatenate(([t0]*(k+1), inner, [tf]*(k+1)))
    assert len(knots) == ncoef + k + 1
    return knots

# ---------------- Synthetic true trajectory (control points) ----------------
t0, tf = 0.0, 80.0  # seconds (longer to show spline behavior)
times_meas = np.linspace(t0, tf, MEAS_N)
m = N_CONTROL
k = BSPLINE_DEGREE
ncoef = m
knots = make_clamped_knots(t0, tf, ncoef, k)

# create some smooth ground-truth control points
cpx_true = np.linspace(-30, 30, ncoef) + 5.0*np.sin(np.linspace(0,3*np.pi,ncoef))
cpy_true = 10.0*np.sin(np.linspace(-1.5,1.5,ncoef)) + np.linspace(-6,6,ncoef)
ctheta_true = np.deg2rad(8.0*np.sin(np.linspace(0,2*np.pi,ncoef)) + np.linspace(0,30,ncoef)/1.5)

spline_px_true = BSpline(knots, cpx_true, k)
spline_py_true = BSpline(knots, cpy_true, k)
spline_th_true = BSpline(knots, ctheta_true, k)

def eval_spline_all(tq, cp_x, cp_y, cp_theta):
    sx = BSpline(knots, cp_x, k)
    sy = BSpline(knots, cp_y, k)
    st = BSpline(knots, cp_theta, k)
    return np.column_stack((sx(tq), sy(tq), st(tq)))

# sample measurement points (body-frame offsets + noise)
# generate true poses at measurement times
poses_true = eval_spline_all(times_meas, cpx_true, cpy_true, ctheta_true)
positions_true = poses_true[:, :2]
# simulate small body-frame sensor offsets/noise (like before)
pos_noisy_body = (np.column_stack((0.2*np.sin(np.linspace(0,6,len(times_meas))), 0.08*np.cos(np.linspace(0,4,len(times_meas)))))
                  + SIG_POS_NOISE * np.random.randn(len(times_meas),2))
Rts_true = [np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]]) for th in poses_true[:,2]]
map_points_noisy = np.array([Rts_true[i] @ pos_noisy_body[i] + positions_true[i] for i in range(len(times_meas))])
z_sample_noisy_positions = bilinear_interpolate_grid(xs, ys, Zg, map_points_noisy[:,0], map_points_noisy[:,1])
z_obs = z_sample_noisy_positions + SIG_Z * np.random.randn(len(z_sample_noisy_positions))

# ---------------- Synthetic IMU generation (higher rate) ----------------
imu_dt = 1.0 / IMU_RATE
imu_times = np.arange(t0, tf+1e-9, imu_dt)
# evaluate true states on imu_times
poses_true_imu = eval_spline_all(imu_times, cpx_true, cpy_true, ctheta_true)
pos_imu = poses_true_imu[:, :2]
th_imu = poses_true_imu[:, 2]
# compute body-frame accelerations by differentiating world-frame positions
# compute first and second derivatives via finite differences
vel_world = np.gradient(pos_imu, imu_dt, axis=0)
acc_world = np.gradient(vel_world, imu_dt, axis=0)  # acceleration in world frame
# convert accelerations to body frame: a_b = R^T * a_world (ignore gravity since in-plane)
acc_body = np.zeros_like(acc_world)
for i in range(len(imu_times)):
    R_i = np.array([[np.cos(th_imu[i]), -np.sin(th_imu[i])],[np.sin(th_imu[i]), np.cos(th_imu[i])]])
    acc_body[i] = R_i.T @ acc_world[i]
# compute angular rates (yaw rate)
yaw_rates = np.gradient(th_imu, imu_dt)
# add measurement noise
yaw_rates_meas = yaw_rates + SIG_IMU_GYRO * np.random.randn(len(yaw_rates))
acc_body_meas = acc_body + SIG_IMU_ACC * np.random.randn(*acc_body.shape)

# Pre-integration between measurement times: for each pair i->i+1 integrate imu samples
def preintegrate_imu_between(imu_t, gyro_meas, acc_meas, ta, tb):
    # mask samples between ta (inclusive) and tb (exclusive of tb maybe include both)
    mask = (imu_t >= ta - 1e-12) & (imu_t <= tb + 1e-12)
    ts = imu_t[mask]
    if len(ts) < 2:
        # fallback: linear interpolation of gyro rate times dt
        dt = tb - ta
        # approximate delta theta
        omega_a = np.interp(ta, imu_t, gyro_meas)
        omega_b = np.interp(tb, imu_t, gyro_meas)
        delta_theta = 0.5*(omega_a + omega_b) * dt
        # approximate delta_p as zero if no samples
        delta_p = np.array([0.0, 0.0])
        return delta_theta, delta_p
    w = gyro_meas[mask]
    a = acc_meas[mask]
    # trapezoidal integration for delta_theta
    delta_theta = np.trapz(w, ts)
    # double integrate accelerations in body frame to get delta position in body frame
    # first integrate acceleration to velocity (v_b), assuming initial velocity unknown; we assume zero initial velocity in preintegration
    # v_b = cumulative integral of a dt
    v_b = np.cumsum((a[:-1] + a[1:]) / 2 * np.diff(ts)[:,None], axis=0)  # length len(ts)-1
    # delta_p = integral of v_b dt over the intervals
    # approximate delta_p by sum v_b * dt, using midpoints
    dt_intervals = np.diff(ts)
    if len(v_b) == 0:
        delta_p = np.array([0.0, 0.0])
    else:
        delta_p = np.sum(v_b * dt_intervals[:,None], axis=0)
    return delta_theta, delta_p

# compute preintegrations for each pair (i -> i+1)
imu_preint_delta_theta = np.zeros(len(times_meas)-1)
imu_preint_delta_p = np.zeros((len(times_meas)-1,2))
for i in range(len(times_meas)-1):
    ta = times_meas[i]; tb = times_meas[i+1]
    dt_theta, dp = preintegrate_imu_between(imu_times, yaw_rates_meas, acc_body_meas, ta, tb)
    imu_preint_delta_theta[i] = dt_theta + 0.0 * np.random.randn()  # could add preint noise if desired
    imu_preint_delta_p[i] = dp + 0.0 * np.random.randn(2)

# ---------------- B-spline basis evaluation matrix ----------------
def compute_basis_matrix(tq):
    B = np.zeros((len(tq), ncoef))
    for j in range(ncoef):
        coeffs = np.zeros(ncoef); coeffs[j]=1.0
        S = BSpline(knots, coeffs, k)
        B[:, j] = S(tq)
    return B

B_meas = compute_basis_matrix(times_meas)
# for imu pairs, basis diff for theta: B(tb)-B(ta)
B_imu_pairs = np.zeros((len(times_meas)-1, ncoef))
for i in range(len(times_meas)-1):
    Bt = BSpline(knots, np.eye(ncoef)[:,0], k)  # dummy reuse not used, we'll compute row-wise
    B_imu_pairs[i,:] = compute_basis_matrix([times_meas[i+1]])[0] - compute_basis_matrix([times_meas[i]])[0]

# ---------------- residuals and jacobian ----------------
def pack_params(cpx, cpy, cth):
    return np.concatenate([cpx, cpy, cth])
def unpack_params(x):
    cpx = x[0:ncoef]; cpy = x[ncoef:2*ncoef]; cth = x[2*ncoef:3*ncoef]; return cpx, cpy, cth

def residuals_and_jac(x):
    cpx, cpy, cth = unpack_params(x)
    px_t = B_meas @ cpx
    py_t = B_meas @ cpy
    th_t = B_meas @ cth
    cos_t = np.cos(th_t); sin_t = np.sin(th_t)
    pts_map_x = cos_t * pos_noisy_body[:,0] - sin_t * pos_noisy_body[:,1] + px_t
    pts_map_y = sin_t * pos_noisy_body[:,0] + cos_t * pos_noisy_body[:,1] + py_t
    fvals = bilinear_interpolate_grid(xs, ys, Zg, pts_map_x, pts_map_y)
    r_map = fvals - z_obs  # MEAS_N
    # IMU rotation residuals: th_{i+1} - th_i - delta_theta_meas
    th_diff = th_t[1:] - th_t[:-1]
    r_imu_rot = (th_diff - imu_preint_delta_theta) * np.sqrt(W_IMU_ROT)
    # IMU translation residuals: R_a^T (p_b - p_a) - delta_p_meas (2D vector)
    m_imu = len(imu_preint_delta_theta)
    r_imu_trans = np.zeros((m_imu, 2))
    # prepare arrays for Jacobian
    m_r = len(r_map); m_rot = len(r_imu_rot); m_trans = m_imu * 2
    J = np.zeros((m_r + m_rot + m_trans + 3*(ncoef-2 if ncoef>2 else 0), 3*ncoef))
    grad_fx, grad_fy = interp_grad_at(xs, ys, dz_dx, dz_dy, pts_map_x, pts_map_y)
    # Map jacobians per basis index
    for j in range(ncoef):
        Nj = B_meas[:, j]  # shape MEAS_N
        # derivative wrt cpx_j and cpy_j
        J[0:m_r, j] = grad_fx * Nj
        J[0:m_r, ncoef + j] = grad_fy * Nj
        # derivative wrt cth_j: use dR/dth * p_body as in earlier code
        dRx = (-sin_t * pos_noisy_body[:,0] - cos_t * pos_noisy_body[:,1])
        dRy = ( cos_t * pos_noisy_body[:,0] - sin_t * pos_noisy_body[:,1])
        J[0:m_r, 2*ncoef + j] = (grad_fx * dRx + grad_fy * dRy) * Nj
    # IMU rotation jacobians: derivative wrt cth_j is (B_{i+1,j} - B_{i,j}) * sqrt(W_IMU_ROT)
    row_rot_start = m_r
    for i in range(m_rot):
        J[row_rot_start + i, 2*ncoef:3*ncoef] = B_imu_pairs[i] * np.sqrt(W_IMU_ROT)
    # IMU translation residuals and jacobians
    row_trans_start = m_r + m_rot
    for i in range(m_imu):
        # indices a=i, b=i+1
        # compute s = p_b - p_a where p = [px_t, py_t]
        s = np.array([px_t[i+1] - px_t[i], py_t[i+1] - py_t[i]])
        # rotation at a
        ca = cos_t[i]; sa = sin_t[i]
        Ra = np.array([[ca, -sa],[sa, ca]])
        RaT = Ra.T
        # residual r_p = RaT @ s - delta_p_meas
        r_p = RaT @ s - imu_preint_delta_p[i]
        r_imu_trans[i] = r_p * np.sqrt(W_IMU_TRANS)
        # fill J rows (two rows) for this residual
        # derivative wrt cpx_j: for p_b -> +RaT * B_meas[b,j]; for p_a -> -RaT * B_meas[a,j]
        for j in range(ncoef):
            Bj_a = B_meas[i, j]
            Bj_b = B_meas[i+1, j]
            # contribution to x and y rows
            # column for cpx_j
            col_px = j
            J[row_trans_start + 2*i + 0, col_px] = RaT[0,0] * (Bj_b - Bj_a) * np.sqrt(W_IMU_TRANS)
            J[row_trans_start + 2*i + 1, col_px] = RaT[1,0] * (Bj_b - Bj_a) * np.sqrt(W_IMU_TRANS)
            # column for cpy_j
            col_py = ncoef + j
            J[row_trans_start + 2*i + 0, col_py] = RaT[0,1] * (Bj_b - Bj_a) * np.sqrt(W_IMU_TRANS)
            J[row_trans_start + 2*i + 1, col_py] = RaT[1,1] * (Bj_b - Bj_a) * np.sqrt(W_IMU_TRANS)
            # derivative wrt cth_j: only affects RaT (through theta_a)
            # dRaT/dtheta = (dR/dtheta)^T evaluated at a
            dR_dth = np.array([[-sa, -ca],[ca, -sa]])
            dRaT = dR_dth.T  # (2x2)
            # contribution = dRaT @ s * B_meas[a,j]
            J[row_trans_start + 2*i + 0, 2*ncoef + j] = (dRaT[0,0]*s[0] + dRaT[0,1]*s[1]) * B_meas[i, j] * np.sqrt(W_IMU_TRANS)
            J[row_trans_start + 2*i + 1, 2*ncoef + j] = (dRaT[1,0]*s[0] + dRaT[1,1]*s[1]) * B_meas[i, j] * np.sqrt(W_IMU_TRANS)
    # place r_imu_trans into residual vector
    r_trans_flat = r_imu_trans.reshape(-1) * 1.0  # already scaled
    # Smoothness residuals as second-difference on control points
    r_smooth = np.zeros(0)
    if W_SMOOTH > 0 and ncoef > 2:
        rsx = (cpx[2:] - 2*cpx[1:-1] + cpx[:-2]) * np.sqrt(W_SMOOTH)
        rsy = (cpy[2:] - 2*cpy[1:-1] + cpy[:-2]) * np.sqrt(W_SMOOTH)
        rst = (cth[2:] - 2*cth[1:-1] + cth[:-2]) * np.sqrt(W_SMOOTH)
        r_smooth = np.concatenate([rsx, rsy, rst])
        # fill jacobian rows for smoothness
        row0 = row_trans_start + m_trans
        idx = 0
        # px block
        for j in range(ncoef-2):
            row = row0 + idx
            J[row, j] = 1.0 * np.sqrt(W_SMOOTH)
            J[row, j+1] = -2.0 * np.sqrt(W_SMOOTH)
            J[row, j+2] = 1.0 * np.sqrt(W_SMOOTH)
            idx += 1
        # py block
        for j in range(ncoef-2):
            row = row0 + idx
            J[row, ncoef + j] = 1.0 * np.sqrt(W_SMOOTH)
            J[row, ncoef + j+1] = -2.0 * np.sqrt(W_SMOOTH)
            J[row, ncoef + j+2] = 1.0 * np.sqrt(W_SMOOTH)
            idx += 1
        # th block
        for j in range(ncoef-2):
            row = row0 + idx
            J[row, 2*ncoef + j] = 1.0 * np.sqrt(W_SMOOTH)
            J[row, 2*ncoef + j+1] = -2.0 * np.sqrt(W_SMOOTH)
            J[row, 2*ncoef + j+2] = 1.0 * np.sqrt(W_SMOOTH)
            idx += 1
    # assemble residual vector r = [r_map; r_imu_rot; r_trans_flat; r_smooth]
    r = np.concatenate([r_map, r_imu_rot, r_trans_flat, r_smooth])
    return r, J

# ---------------- initial guess ----------------
cpx_init = cpx_true + 1.0 * np.random.randn(ncoef)
cpy_init = cpy_true + 1.0 * np.random.randn(ncoef)
cth_init = ctheta_true + np.deg2rad(3.0) * np.random.randn(ncoef)
x0 = pack_params(cpx_init, cpy_init, cth_init)

# ---------------- run optimization ----------------
print("Optimizing spline control points with IMU pre-integration residuals...")
res = least_squares(lambda x: residuals_and_jac(x)[0], x0, jac=lambda x: residuals_and_jac(x)[1],
                    method='lm', xtol=1e-8, ftol=1e-8, gtol=1e-8, max_nfev=200)
print("Done. success:", res.success, "message:", res.message)
x_opt = res.x
cpx_opt, cpy_opt, cth_opt = unpack_params(x_opt)

# ---------------- evaluate results ----------------
poses_init = eval_spline_all(times_meas, cpx_init, cpy_init, cth_init)
poses_opt = eval_spline_all(times_meas, cpx_opt, cpy_opt, cth_opt)
poses_true = poses_true
pos_err_init = np.linalg.norm(poses_init[:,:2] - poses_true[:,:2], axis=1)
pos_err_opt = np.linalg.norm(poses_opt[:,:2] - poses_true[:,:2], axis=1)
yaw_err_init = np.abs(np.unwrap(poses_init[:,2]) - np.unwrap(poses_true[:,2]))
yaw_err_opt = np.abs(np.unwrap(poses_opt[:,2]) - np.unwrap(poses_true[:,2]))

print(f"Position RMSE initial: {np.sqrt(np.mean(pos_err_init**2)):.3f}  optimized: {np.sqrt(np.mean(pos_err_opt**2)):.3f}")
print(f"Yaw RMSE initial(deg): {np.rad2deg(np.sqrt(np.mean(yaw_err_init**2))):.3f}  optimized: {np.rad2deg(np.sqrt(np.mean(yaw_err_opt**2))):.3f}")

# ---------------- visualizations ----------------
fig, ax = plt.subplots(1,2,figsize=(16,6))
ax[0].contour(xs, ys, Zg, levels=30, linewidths=0.6)
ax[0].plot(poses_true[:,0], poses_true[:,1], 'k-', label='true')
ax[0].plot(poses_init[:,0], poses_init[:,1], 'r--', label='init')
ax[0].plot(poses_opt[:,0], poses_opt[:,1], 'g-', label='opt')
ax[0].scatter(map_points_noisy[:,0], map_points_noisy[:,1], c='orange', s=8, label='samples')
ax[0].legend(); ax[0].set_title("Trajectories on map"); ax[0].set_xlabel("X"); ax[0].set_ylabel("Y")

ax[1].plot(times_meas, pos_err_init, label='pos err init')
ax[1].plot(times_meas, pos_err_opt, label='pos err opt')
ax[1].plot(times_meas, np.rad2deg(yaw_err_init), '--', label='yaw err init (deg)')
ax[1].plot(times_meas, np.rad2deg(yaw_err_opt), '-', label='yaw err opt (deg)')
ax[1].legend(); ax[1].set_title("Per-sample errors"); ax[1].set_xlabel("time (s)"); ax[1].grid(True)
plt.show()

# residual histograms
r_init, _ = residuals_and_jac(x0)
r_opt, _ = residuals_and_jac(x_opt)
m_r = MEAS_N
plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.hist(r_init[:m_r], bins=30); plt.title("Map residuals init")
plt.subplot(1,2,2); plt.hist(r_opt[:m_r], bins=30); plt.title("Map residuals opt")
plt.show()

# theta control points
plt.figure(figsize=(8,4))
plt.plot(np.arange(ncoef), ctheta_true, label='true theta ctrl pts')
plt.plot(np.arange(ncoef), cth_init, 'r--', label='init theta ctrl pts')
plt.plot(np.arange(ncoef), cth_opt, 'g-', label='opt theta ctrl pts')
plt.legend(); plt.title("Theta control points (true / init / opt)")
plt.show()

# parameter summary
print("\nParameters summary:")
print(f"BSPLINE_DEGREE={BSPLINE_DEGREE}, N_CONTROL={N_CONTROL}, MEAS_N={MEAS_N}, IMU_RATE={IMU_RATE}")
print(f"SIG_Z={SIG_Z}, SIG_IMU_GYRO={SIG_IMU_GYRO}, SIG_IMU_ACC={SIG_IMU_ACC}")
print(f"W_IMU_ROT={W_IMU_ROT}, W_IMU_TRANS={W_IMU_TRANS}, W_SMOOTH={W_SMOOTH}")
print("To tune behavior, adjust these variables at the top of the script and re-run.")

