# Particle filter SITAN demo (2D) - Bootstrap particle filter on position error (dx,dy)
# - Uses same map & trajectory generation idea as previous demo
# - Particles represent position error: e = nominal - true (so true = nominal - e)
# - Proposal: Random-walk process model for errors; measurement likelihood from gravimeter scalar map
# - Resampling: systematic resampling; effective N gating
#
# Dependencies: numpy, scipy, matplotlib
# Run time: moderate. You can reduce n_particles for speed (e.g., 500 -> faster).
#
# Outputs: trajectory plot, error vs time, RMSE numbers, sample particle spread animation-like quiver at selected times

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
import time

np.random.seed(3)

# --- Simulation parameters (same-ish as previous) ---
dt = 0.1
T = 200.0
steps = int(T / dt)
times = np.arange(0, steps) * dt

# Map area and grid
xmin, xmax, ymin, ymax = -250.0, 250.0, -250.0, 250.0
grid_res = 2.0
xs = np.arange(xmin, xmax + grid_res, grid_res)
ys = np.arange(ymin, ymax + grid_res, grid_res)
XX, YY = np.meshgrid(xs, ys, indexing='xy')

g0 = 9.80665
anomalies = np.zeros_like(XX)
bumps = [
    ( 60,  80,  3e-5, 40.0),
    (-90, -50, -2e-5, 60.0),
    (  0,   0,  4e-5, 30.0),
    (120, -120, 1.5e-5, 50.0)
]
for cx, cy, amp, sigma in bumps:
    anomalies += amp * np.exp(-(((XX - cx)**2 + (YY - cy)**2) / (2 * sigma**2)))
anomalies = gaussian_filter(anomalies, sigma=1.0)
g_map = g0 + anomalies

g_interp = RegularGridInterpolator((xs, ys), g_map.T, bounds_error=False, fill_value=None)
dgdy, dgdx = np.gradient(g_map, grid_res, grid_res)  # for diagnostics if needed
dgdx_interp = RegularGridInterpolator((xs, ys), dgdx.T, bounds_error=False, fill_value=0.0)
dgdy_interp = RegularGridInterpolator((xs, ys), dgdy.T, bounds_error=False, fill_value=0.0)

# True trajectory (same functional form)
theta_path = np.linspace(0, 4*np.pi, steps)
r = 120 + 60 * np.sin(0.5 * theta_path)
true_x = r * np.cos(theta_path)
true_y = r * np.sin(theta_path) + 30 * np.sin(0.2 * theta_path)
true_vx = np.gradient(true_x, dt)
true_vy = np.gradient(true_y, dt)
true_yaw = np.arctan2(true_vy, true_vx)
true_yaw = gaussian_filter(true_yaw, sigma=2)
true_ax = np.gradient(true_vx, dt)
true_ay = np.gradient(true_vy, dt)

# IMU sim (similar)
accel_noise_std = 0.02
gyro_noise_std = 0.001
accel_bias_true = np.array([0.02, -0.01, 0.0])
gyro_bias_true = 0.002

accel_meas = np.zeros((steps, 3))
gyro_meas = np.zeros(steps)
gravimeter_meas = np.zeros(steps)

for k in range(steps):
    xk = true_x[k]; yk = true_y[k]
    g_at = g_interp((xk, yk))
    a_nav = np.array([true_ax[k], true_ay[k], 0.0])
    specific_force_nav = a_nav - np.array([0,0,-g_at])
    yaw = true_yaw[k]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw),  np.cos(yaw), 0],
                     [0,             0,          1]])
    f_b_true = C_nb.T.dot(specific_force_nav)
    accel_meas[k] = f_b_true + accel_bias_true + np.random.randn(3) * accel_noise_std
    if k == 0:
        yaw_rate = 0.0
    else:
        yaw_rate = (true_yaw[k] - true_yaw[k-1]) / dt
    gyro_meas[k] = yaw_rate + gyro_bias_true + np.random.randn() * gyro_noise_std
    gravimeter_meas[k] = g_at + np.random.randn() * 5e-6  # baseline noise

# INS dead-reckon (same as before)
ins_x = np.zeros(steps); ins_y = np.zeros(steps)
ins_vx = np.zeros(steps); ins_vy = np.zeros(steps)
ins_yaw = np.zeros(steps)
ins_x[0] = true_x[0] + 5.0; ins_y[0] = true_y[0] - 3.0
ins_vx[0] = true_vx[0]; ins_vy[0] = true_vy[0]
ins_yaw[0] = true_yaw[0] + 0.05

for k in range(1, steps):
    ins_yaw[k] = ins_yaw[k-1] + gyro_meas[k-1] * dt
    f_b = accel_meas[k-1]
    yaw = ins_yaw[k-1]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw),  np.cos(yaw), 0],
                     [0,             0,          1]])
    specific_force_nav = C_nb.dot(f_b)
    ax = specific_force_nav[0]; ay = specific_force_nav[1]
    ins_vx[k] = ins_vx[k-1] + ax * dt
    ins_vy[k] = ins_vy[k-1] + ay * dt
    ins_x[k] = ins_x[k-1] + ins_vx[k-1] * dt + 0.5 * ax * dt**2
    ins_y[k] = ins_y[k-1] + ins_vy[k-1] * dt + 0.5 * ay * dt**2

# --- Particle Filter setup ---
n_particles = 1500  # adjust for speed/accuracy
# Particles represent error e = nominal - true; initialize around initial INS error
init_error = np.array([ins_x[0] - true_x[0], ins_y[0] - true_y[0]])
particles = np.random.randn(n_particles, 2) * np.array([5.0, 5.0]) + init_error  # large spread initially
weights = np.ones(n_particles) / n_particles

# Process noise for error evolution (random-walk)
proc_std = np.array([0.4, 0.4])  # meters per step (tune this)
# Measurement noise (gravimeter + map)
meas_std = 5e-6
R_var = meas_std**2 + (2e-6)**2  # include small map uncertainty

# Resampling threshold
neff_threshold = 0.5 * n_particles

# Storage for PF estimate
pf_est = np.zeros((steps, 2))
pf_var = np.zeros((steps, 2))
ess_record = np.zeros(steps)
resample_count = 0

start_time = time.time()

for k in range(steps):
    # Predict particle evolution: simple random-walk on error
    if k > 0:
        particles += np.random.randn(n_particles, 2) * proc_std

    # At measurement times (gravimeter available every step in this sim), compute weights
    z = gravimeter_meas[k]
    # Compute hypothesized true positions for each particle: r = ins_pos - e
    ins_pos = np.array([ins_x[k], ins_y[k]])
    pts = ins_pos.reshape(1,2) - particles  # shape (n_particles, 2)
    # Clip pts to map bounds to avoid NaNs
    pts_clipped = np.empty_like(pts)
    pts_clipped[:,0] = np.clip(pts[:,0], xmin+1e-3, xmax-1e-3)
    pts_clipped[:,1] = np.clip(pts[:,1], ymin+1e-3, ymax-1e-3)
    g_preds = g_interp(pts_clipped)  # vectorized interpolation

    # Likelihood under Gaussian noise
    residuals = z - g_preds  # array length n_particles
    logw = -0.5 * (residuals**2) / R_var
    # normalize in log-space for numerical stability
    logw -= np.max(logw)
    w = np.exp(logw)
    weights = w / np.sum(w)

    # Compute effective sample size
    ess = 1.0 / np.sum(weights**2)
    ess_record[k] = ess

    # Estimate state: weighted mean of particles (error e)
    mean_e = np.sum(particles * weights.reshape(-1,1), axis=0)
    var_e = np.sum((particles - mean_e)**2 * weights.reshape(-1,1), axis=0)
    pf_est[k] = mean_e
    pf_var[k] = var_e

    # Resample if necessary (systematic resampling)
    if ess < neff_threshold:
        # systematic resample indices
        positions = (np.arange(n_particles) + np.random.rand()) / n_particles
        cumulative = np.cumsum(weights)
        idx = np.searchsorted(cumulative, positions)
        particles = particles[idx].copy()
        weights.fill(1.0 / n_particles)
        # add small jitter to diversify
        particles += np.random.randn(n_particles, 2) * (proc_std * 0.2)
        resample_count += 1

# Compute corrected trajectory: corr = ins - pf_est
corr_x_pf = ins_x - pf_est[:,0]
corr_y_pf = ins_y - pf_est[:,1]

pf_err = np.sqrt((corr_x_pf - true_x)**2 + (corr_y_pf - true_y)**2)
ins_err = np.sqrt((ins_x - true_x)**2 + (ins_y - true_y)**2)

end_time = time.time()

# Results
rmse_ins = np.sqrt(np.mean(ins_err**2))
rmse_pf = np.sqrt(np.mean(pf_err**2))
print(f"Particles: {n_particles}, proc_std={proc_std}, meas_std={meas_std}")
print(f"RMSE INS: {rmse_ins:.2f} m, RMSE ParticleFilter-corrected: {rmse_pf:.2f} m")
print(f"ESS mean: {np.mean(ess_record):.1f}, resamples: {resample_count}, runtime: {end_time-start_time:.1f}s")

# Plot trajectories
plt.figure(figsize=(8,6))
plt.contourf(XX, YY, (g_map-g0)*1e5, levels=40)
plt.colorbar(label='Gravity anomaly (1e-5 m/s^2)')
plt.plot(true_x, true_y, 'y-', lw=2, label='True')
plt.plot(ins_x, ins_y, 'C1--', label='INS dead-reckon')
plt.plot(corr_x_pf, corr_y_pf, 'C2-', lw=1.5, label='PF-corrected')
plt.legend(); plt.xlabel('X (m)'); plt.ylabel('Y (m)'); plt.title('Particle Filter SITAN (trajectory)'); plt.grid(True); plt.show()

# Plot error vs time
plt.figure(figsize=(9,4))
plt.plot(times, ins_err, label='INS error (m)', alpha=0.7)
plt.plot(times, pf_err, label='PF corrected error (m)', alpha=0.8)
plt.xlabel('Time (s)'); plt.ylabel('Position error (m)'); plt.legend(); plt.grid(True); plt.title('Position error over time')
plt.show()

# Visualize particle cloud at a few sample times
sample_times = [int(0.05*steps), int(0.25*steps), int(0.5*steps), int(0.75*steps)]
for t in sample_times:
    plt.figure(figsize=(6,5))
    plt.contourf(XX, YY, (g_map-g0)*1e5, levels=40)
    pts = np.column_stack((ins_x[t] - particles[:,0], ins_y[t] - particles[:,1]))
    plt.scatter(pts[:,0], pts[:,1], s=4, alpha=0.3, label='particles (hypotheses)')
    plt.plot(true_x[t], true_y[t], 'yo', label='true')
    plt.plot(ins_x[t], ins_y[t], 'C1x', label='ins')
    plt.plot(corr_x_pf[t], corr_y_pf[t], 'C2s', label='pf_est')
    plt.legend(); plt.title(f'Particle cloud at t={t*dt:.1f}s'); plt.xlim(ins_x[t]-80, ins_x[t]+80); plt.ylim(ins_y[t]-80, ins_y[t]+80)
    plt.show()

# show a couple of diagnostics
print("Sample of final estimated error (m):", pf_est[-1])
print("Sample variance final (m^2):", pf_var[-1])
