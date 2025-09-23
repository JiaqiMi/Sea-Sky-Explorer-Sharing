# PF + Sliding-window MAP with analytic Jacobian (2D SITAN demo)
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import least_squares
import time

np.random.seed(5)

# -------- simulation params (same style as yours) ----------
dt = 0.1
T = 200.0
steps = int(T / dt)
times = np.arange(steps) * dt

xmin, xmax, ymin, ymax = -250.0, 250.0, -250.0, 250.0
grid_res = 2.0
xs = np.arange(xmin, xmax+grid_res, grid_res)
ys = np.arange(ymin, ymax+grid_res, grid_res)
XX, YY = np.meshgrid(xs, ys, indexing='xy')
g0 = 9.80665

# gravity map (slightly stronger anomalies to help observability)
anomalies = np.zeros_like(XX)
bumps = [(60,80,3e-5,40.0), (-90,-50,-2e-5,60.0), (0,0,4e-5,30.0), (120,-120,1.5e-5,50.0)]
for cx,cy,amp,sigma in bumps:
    anomalies += amp * np.exp(-(((XX-cx)**2 + (YY-cy)**2)/(2*sigma**2)))
anomalies = gaussian_filter(anomalies, sigma=1.0)
g_map = g0 + anomalies

g_interp = RegularGridInterpolator((xs, ys), g_map.T, bounds_error=False, fill_value=None)
dgdy, dgdx = np.gradient(g_map, grid_res, grid_res)
dgdx_interp = RegularGridInterpolator((xs, ys), dgdx.T, bounds_error=False, fill_value=0.0)
dgdy_interp = RegularGridInterpolator((xs, ys), dgdy.T, bounds_error=False, fill_value=0.0)

# true trajectory
theta_path = np.linspace(0, 4*np.pi, steps)
r = 120 + 60*np.sin(0.5*theta_path)
true_x = r*np.cos(theta_path)
true_y = r*np.sin(theta_path) + 30*np.sin(0.2*theta_path)
true_vx = np.gradient(true_x, dt)
true_vy = np.gradient(true_y, dt)
true_yaw = np.arctan2(true_vy, true_vx)
true_yaw = gaussian_filter(true_yaw, sigma=2)
true_ax = np.gradient(true_vx, dt)
true_ay = np.gradient(true_vy, dt)

# IMU sim
accel_noise_std = 0.02
gyro_noise_std = 0.001
accel_bias_true = np.array([0.02, -0.01, 0.0])
gyro_bias_true = 0.002

accel_meas = np.zeros((steps,3)); gyro_meas = np.zeros(steps); gravimeter_meas = np.zeros(steps)
for k in range(steps):
    xk, yk = true_x[k], true_y[k]
    g_at = g_interp((xk, yk))
    a_nav = np.array([true_ax[k], true_ay[k], 0.0])
    specific_force_nav = a_nav - np.array([0,0,-g_at])
    yaw = true_yaw[k]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw), np.cos(yaw), 0],[0,0,1]])
    f_b_true = C_nb.T.dot(specific_force_nav)
    accel_meas[k] = f_b_true + accel_bias_true + np.random.randn(3)*accel_noise_std
    if k==0: yaw_rate=0.0
    else: yaw_rate=(true_yaw[k]-true_yaw[k-1])/dt
    gyro_meas[k] = yaw_rate + gyro_bias_true + np.random.randn()*gyro_noise_std
    gravimeter_meas[k] = g_at + np.random.randn()*5e-6

# INS dead-reckon
ins_x = np.zeros(steps); ins_y = np.zeros(steps)
ins_vx = np.zeros(steps); ins_vy = np.zeros(steps); ins_yaw = np.zeros(steps)
ins_x[0] = true_x[0]+5.0; ins_y[0]=true_y[0]-3.0
ins_vx[0]=true_vx[0]; ins_vy[0]=true_vy[0]; ins_yaw[0]=true_yaw[0]+0.05
for k in range(1, steps):
    ins_yaw[k] = ins_yaw[k-1] + gyro_meas[k-1]*dt
    f_b = accel_meas[k-1]; yaw = ins_yaw[k-1]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],[np.sin(yaw), np.cos(yaw), 0],[0,0,1]])
    specific_nav = C_nb.dot(f_b)
    ax, ay = specific_nav[0], specific_nav[1]
    ins_vx[k] = ins_vx[k-1] + ax*dt; ins_vy[k] = ins_vy[k-1] + ay*dt
    ins_x[k] = ins_x[k-1] + ins_vx[k-1]*dt + 0.5*ax*dt**2
    ins_y[k] = ins_y[k-1] + ins_vy[k-1]*dt + 0.5*ay*dt**2

# ------------- Particle Filter (bootstrap) -------------
n_particles = 1000
init_error = np.array([ins_x[0]-true_x[0], ins_y[0]-true_y[0]])
particles = np.random.randn(n_particles,2) * np.array([8.0, 8.0]) + init_error
weights = np.ones(n_particles)/n_particles
proc_std = np.array([0.6, 0.6])
meas_std = 5e-6
R_var = meas_std**2 + (2e-6)**2
neff_thresh = 0.5*n_particles

pf_est = np.zeros((steps,2))
t0 = time.time()
for k in range(steps):
    if k>0:
        particles += np.random.randn(n_particles,2) * proc_std
    z = gravimeter_meas[k]
    ins_pos = np.array([ins_x[k], ins_y[k]])
    pts = ins_pos.reshape(1,2) - particles
    pts[:,0] = np.clip(pts[:,0], xmin+1e-3, xmax-1e-3); pts[:,1] = np.clip(pts[:,1], ymin+1e-3, ymax-1e-3)
    g_preds = g_interp(pts)
    res = z - g_preds
    logw = -0.5*(res**2)/R_var
    logw -= np.max(logw)
    w = np.exp(logw); weights = w/np.sum(w)
    mean_e = np.sum(particles * weights.reshape(-1,1), axis=0)
    pf_est[k] = mean_e
    ess = 1.0/np.sum(weights**2)
    if ess < neff_thresh:
        positions = (np.arange(n_particles) + np.random.rand())/n_particles
        cumulative = np.cumsum(weights)
        idx = np.searchsorted(cumulative, positions)
        particles = particles[idx].copy()
        weights.fill(1.0/n_particles)
        particles += np.random.randn(n_particles,2) * (proc_std*0.18)
t_pf = time.time()-t0

# PF corrected
corr_x_pf = ins_x - pf_est[:,0]; corr_y_pf = ins_y - pf_est[:,1]

# ------------- Sliding-window MAP smoothing (analytic jacobian) -------------
# window params (tuneable)
window_sec = 12.0
window_size = int(window_sec / dt)
stride = window_size // 2
sigma_g = 5e-6
sigma_proc = 0.5
sigma_anchor = 2.0

# initial smoothed_e = pf_est
smoothed_e = pf_est.copy()

def residual_and_jac(e_vec, ins_pos_window, z_window, pf_center_est):
    W = len(z_window)
    e = e_vec.reshape((W,2))
    # gravity residuals and gradient
    pts = ins_pos_window - e
    pts[:,0] = np.clip(pts[:,0], xmin+1e-3, xmax-1e-3); pts[:,1] = np.clip(pts[:,1], ymin+1e-3, ymax-1e-3)
    g_preds = g_interp(pts)
    rg = (g_preds - z_window) / sigma_g  # (W,)
    # build residual vector
    # smoothness residuals (W-1)*2
    diffs = (e[1:,:] - e[:-1,:]).ravel() / sigma_proc
    # anchor residual (2,)
    center_idx = W//2
    r_anchor = (e[center_idx,:] - pf_center_est) / sigma_anchor
    res = np.concatenate([rg, diffs, r_anchor])
    # Now analytic Jacobian: rows x cols where cols = 2W
    m = res.size; n = 2*W
    J = np.zeros((m, n))
    # Jacobian of rg wrt e_k: drg/de_k = - (dgdx,dgdy)/sigma_g for that k
    # compute grads at pts
    grad_x = dgdx_interp(pts)  # array length W
    grad_y = dgdy_interp(pts)
    for i in range(W):
        J[i, 2*i]   = -grad_x[i] / sigma_g
        J[i, 2*i+1] = -grad_y[i] / sigma_g
    # smoothness jacobian: for each diff index j (corresponds to residual rows W + 2*j ..)
    # index offset
    row = W
    for j in range(W-1):
        # diff residuals order: [dx_j, dy_j] (they are flattened by row)
        # but we used ravel() with row-major, so diffs entries are [e[1,0]-e[0,0], e[1,1]-e[0,1], e[2,0]-e[1,0], ...]
        # handle per coordinate
        # dx part
        J[row, 2*j]     = -1.0 / sigma_proc
        J[row, 2*(j+1)] =  1.0 / sigma_proc
        row += 1
        # dy part
        J[row, 2*j+1]     = -1.0 / sigma_proc
        J[row, 2*(j+1)+1] =  1.0 / sigma_proc
        row += 1
    # anchor jacobian (2 rows)
    J[row, 2*center_idx]   = 1.0 / sigma_anchor
    J[row, 2*center_idx+1] = 0.0
    row += 1
    J[row, 2*center_idx]   = 0.0
    J[row, 2*center_idx+1] = 1.0 / sigma_anchor
    return res, J

# sliding windows
t_s = time.time()
k = 0
while k < steps:
    i0 = k
    i1 = min(k + window_size, steps)
    W = i1 - i0
    if W < 8: break
    ins_pos_window = np.column_stack((ins_x[i0:i1], ins_y[i0:i1]))
    z_window = gravimeter_meas[i0:i1]
    e0 = smoothed_e[i0:i1].ravel()
    pf_center = smoothed_e[i0 + W//2]
    # least squares with analytic jac
    fun = lambda ev: residual_and_jac(ev, ins_pos_window, z_window, pf_center)
    # method 'lm' can be fast but requires m>=n, which is true here; we'll use 'trf' to be robust with bounds
    sol = least_squares(lambda ev: residual_and_jac(ev, ins_pos_window, z_window, pf_center)[0],
                        e0, jac=lambda ev: residual_and_jac(ev, ins_pos_window, z_window, pf_center)[1],
                        method='trf', xtol=1e-6, ftol=1e-6, max_nfev=200)
    e_opt = sol.x.reshape((W,2))
    # commit center half
    commit_start = i0 + W//4
    commit_end = i0 + 3*W//4
    commit_start = max(commit_start, i0)
    commit_end = min(commit_end, i1)
    smoothed_e[commit_start:commit_end] = e_opt[(commit_start-i0):(commit_end-i0)]
    k += stride
t_smooth = time.time()-t_s

# corrected trajectories and errors
corr_x_smooth = ins_x - smoothed_e[:,0]; corr_y_smooth = ins_y - smoothed_e[:,1]
ins_err = np.sqrt((ins_x-true_x)**2 + (ins_y-true_y)**2)
pf_err = np.sqrt((corr_x_pf - true_x)**2 + (corr_y_pf - true_y)**2)
smooth_err = np.sqrt((corr_x_smooth - true_x)**2 + (corr_y_smooth - true_y)**2)

print("Runtimes: PF {:.2f}s, smoothing {:.2f}s".format(t_pf, t_smooth))
print("RMSE: INS {:.2f} m, PF {:.2f} m, PF+MAP {:.2f} m".format(
    np.sqrt(np.mean(ins_err**2)), np.sqrt(np.mean(pf_err**2)), np.sqrt(np.mean(smooth_err**2))
))

# plotting
plt.figure(figsize=(8,6))
plt.contourf(XX, YY, (g_map-g0)*1e5, levels=40)
plt.plot(true_x, true_y, 'y-', lw=2, label='True')
plt.plot(ins_x, ins_y, 'C1--', label='INS')
plt.plot(corr_x_pf, corr_y_pf, 'C2-', label='PF')
plt.plot(corr_x_smooth, corr_y_smooth, 'k-', label='PF+MAP')
plt.legend(); plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.title('PF + Sliding-window MAP'); plt.grid(True); plt.show()

plt.figure(figsize=(9,4))
plt.plot(times, ins_err, label='INS')
plt.plot(times, pf_err, label='PF')
plt.plot(times, smooth_err, label='PF+MAP')
plt.legend(); plt.xlabel('Time (s)'); plt.ylabel('Position error (m)'); plt.grid(True); plt.show()
