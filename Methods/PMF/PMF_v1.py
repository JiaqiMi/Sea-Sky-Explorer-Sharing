# Particle filter replacement for SITAN EKF in the provided demo.
# Copy this entire cell and run (requires numpy, scipy, matplotlib, pandas already used above).

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

np.random.seed(2)

# --- Simulation parameters (same as your original) ---
dt = 0.1
T = 200.0
steps = int(T / dt)
times = np.arange(0, steps) * dt

xmin, xmax, ymin, ymax = -250.0, 250.0, -250.0, 250.0
grid_res = 2.0
xs = np.arange(xmin, xmax + grid_res, grid_res)
ys = np.arange(ymin, ymax + grid_res, grid_res)
XX, YY = np.meshgrid(xs, ys, indexing='xy')

g0 = 9.80665

# Create anomalies (same bumps)
anomalies = np.zeros_like(XX)
bumps = [
    ( 60,  80,  0.00002, 40.0),
    (-90, -50, -0.000015, 60.0),
    (  0,   0,  0.00003, 30.0),
    (120, -120, 0.00001, 50.0)
]
for cx, cy, amp, sigma in bumps:
    anomalies += amp * np.exp(-(((XX - cx)**2 + (YY - cy)**2) / (2 * sigma**2)))
anomalies = gaussian_filter(anomalies, sigma=1.0)
g_map = g0 + anomalies

# Gradients and interpolators
dy, dx = np.gradient(g_map, grid_res, grid_res)
g_interp = RegularGridInterpolator((xs, ys), g_map.T, bounds_error=False, fill_value=None)
dgdx_interp = RegularGridInterpolator((xs, ys), dx.T, bounds_error=False, fill_value=0.0)
dgdy_interp = RegularGridInterpolator((xs, ys), dy.T, bounds_error=False, fill_value=0.0)

# --- True trajectory (same) ---
speed = 2.0
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

# --- IMU simulation (same) ---
accel_noise_std = 0.02
gyro_noise_std = 0.001
accel_bias_true = np.array([0.02, -0.01, 0.0])
gyro_bias_true = 0.002

accel_meas = np.zeros((steps, 3))
gyro_meas = np.zeros(steps)
gravimeter_meas = np.zeros(steps)
grav_noise_std = 5e-6

for k in range(steps):
    xk = true_x[k]
    yk = true_y[k]
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
    gravimeter_meas[k] = g_at + np.random.randn() * grav_noise_std

# --- INS dead-reckoning (same) ---
ins_x = np.zeros(steps)
ins_y = np.zeros(steps)
ins_vx = np.zeros(steps)
ins_vy = np.zeros(steps)
ins_yaw = np.zeros(steps)

ins_x[0] = true_x[0] + 5.0 * 0.01
ins_y[0] = true_y[0] - 3.0 * 0.01
ins_vx[0] = true_vx[0]
ins_vy[0] = true_vy[0]
ins_yaw[0] = true_yaw[0] + 0.05

for k in range(1, steps):
    ins_yaw[k] = ins_yaw[k-1] + (gyro_meas[k-1]) * dt
    f_b = accel_meas[k-1]
    yaw = ins_yaw[k-1]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw),  np.cos(yaw), 0],
                     [0,             0,          1]])
    specific_force_nav = C_nb.dot(f_b)
    ax = specific_force_nav[0]
    ay = specific_force_nav[1]
    ins_vx[k] = ins_vx[k-1] + ax * dt
    ins_vy[k] = ins_vy[k-1] + ay * dt
    ins_x[k] = ins_x[k-1] + ins_vx[k-1] * dt + 0.5 * ax * dt**2
    ins_y[k] = ins_y[k-1] + ins_vy[k-1] * dt + 0.5 * ay * dt**2

# --- Particle Filter implementation ---
# Particle state: [x, y, vx, vy, yaw, bgrav]
Np = 1500  # number of particles (adjustable)
particles = np.zeros((Np, 6))
weights = np.ones(Np) / Np

# initialize particles around INS initial state (small spread)
init_pos_sigma = 5.0  # meters
init_vel_sigma = 0.5  # m/s
init_yaw_sigma = 0.1  # rad
init_bgrav_sigma = 1e-5  # m/s^2

particles[:, 0] = ins_x[0] + np.random.randn(Np) * init_pos_sigma
particles[:, 1] = ins_y[0] + np.random.randn(Np) * init_pos_sigma
particles[:, 2] = ins_vx[0] + np.random.randn(Np) * init_vel_sigma
particles[:, 3] = ins_vy[0] + np.random.randn(Np) * init_vel_sigma
particles[:, 4] = ins_yaw[0] + np.random.randn(Np) * init_yaw_sigma
particles[:, 5] = np.random.randn(Np) * init_bgrav_sigma

# storage for corrected (PF) estimates
pf_x = np.zeros(steps)
pf_y = np.zeros(steps)
pf_vx = np.zeros(steps)
pf_vy = np.zeros(steps)
pf_yaw = np.zeros(steps)
pf_bgrav = np.zeros(steps)
pf_x[0] = np.average(particles[:,0], weights=weights)
pf_y[0] = np.average(particles[:,1], weights=weights)
pf_vx[0] = np.average(particles[:,2], weights=weights)
pf_vy[0] = np.average(particles[:,3], weights=weights)
pf_yaw[0] = np.angle(np.sum(np.exp(1j*particles[:,4]) * weights)) if True else np.average(particles[:,4], weights=weights)
pf_bgrav[0] = np.average(particles[:,5], weights=weights)

# process noise parameters for particle propagation
proc_pos_noise = 0.05  # m
proc_vel_noise = 0.02  # m/s
proc_yaw_noise = 0.005  # rad
proc_bgrav_noise = 1e-7  # m/s^2 (random walk)

# Measurement noise variance (same notion as EKF)
R_grav_pf = grav_noise_std**2 + (1e-6)**2

def systematic_resample(weights):
    N = len(weights)
    positions = (np.arange(N) + np.random.rand()) / N
    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def effective_n(weights):
    return 1.0 / np.sum(np.square(weights))

meas_interval_steps = int(1.0 / dt)
resample_threshold = 0.5 * Np

for k in range(1, steps):
    # --- Propagate particles using IMU (accel_meas[k-1], gyro_meas[k-1]) ---
    gyro_km1 = gyro_meas[k-1]
    f_b_km1 = accel_meas[k-1]  # measured specific force in body
    # propagate yaw
    particles[:, 4] += gyro_km1 * dt + np.random.randn(Np) * proc_yaw_noise
    # compute acceleration in nav for each particle using its yaw and measured body accel
    yaws = particles[:, 4]
    cos_y = np.cos(yaws)
    sin_y = np.sin(yaws)
    # rotate f_b to nav: a_nav ≈ C_nb * f_b (3-vector). We only need x,y components.
    # for planar rotation:
    ax_nav = cos_y * f_b_km1[0] - sin_y * f_b_km1[1]  # approximate
    ay_nav = sin_y * f_b_km1[0] + cos_y * f_b_km1[1]
    # integrate velocity and position with small process noise
    particles[:, 2] += ax_nav * dt + np.random.randn(Np) * proc_vel_noise
    particles[:, 3] += ay_nav * dt + np.random.randn(Np) * proc_vel_noise
    particles[:, 0] += particles[:, 2] * dt + 0.5 * ax_nav * dt**2 + np.random.randn(Np) * proc_pos_noise
    particles[:, 1] += particles[:, 3] * dt + 0.5 * ay_nav * dt**2 + np.random.randn(Np) * proc_pos_noise
    # gravimeter bias random walk
    particles[:, 5] += np.random.randn(Np) * proc_bgrav_noise

    # --- Measurement update at gravimeter rate ---
    if (k % meas_interval_steps) == 0:
        z = gravimeter_meas[k]
        # evaluate predicted gravity at particles' positions
        # prepare array of query points (x,y)
        pts = np.vstack((particles[:,0], particles[:,1])).T
        g_preds = g_interp(pts)
        # predicted meas = g_pred + bgrav_particle
        meas_preds = g_preds + particles[:,5]
        # compute weights via Gaussian likelihood
        # To avoid underflow, compute log-weights then exponentiate after subtracting max
        residuals = z - meas_preds  # shape (Np,)
        log_w = -0.5 * (residuals**2) / R_grav_pf
        # normalize
        log_w = log_w - np.max(log_w)
        w = np.exp(log_w)
        w = w + 1e-300  # avoid zeros
        w = w / np.sum(w)
        weights = w

        # compute weighted estimate
        pf_x[k] = np.sum(weights * particles[:,0])
        pf_y[k] = np.sum(weights * particles[:,1])
        pf_vx[k] = np.sum(weights * particles[:,2])
        pf_vy[k] = np.sum(weights * particles[:,3])
        # for yaw, use circular mean
        sin_y = np.sin(particles[:,4])
        cos_y = np.cos(particles[:,4])
        mean_yaw = np.arctan2(np.sum(weights * sin_y), np.sum(weights * cos_y))
        pf_yaw[k] = mean_yaw
        pf_bgrav[k] = np.sum(weights * particles[:,5])

        # compute effective N
        Neff = effective_n(weights)
        if Neff < resample_threshold:
            idx = systematic_resample(weights)
            particles = particles[idx].copy()
            weights = np.ones(Np) / Np
        # after resampling, we keep particles as-is (they already incorporate uncertainty)
    else:
        # no new measurement: report predicted (unweighted) mean (or previous)
        pf_x[k] = np.mean(particles[:,0])
        pf_y[k] = np.mean(particles[:,1])
        pf_vx[k] = np.mean(particles[:,2])
        pf_vy[k] = np.mean(particles[:,3])
        sin_y = np.sin(particles[:,4])
        cos_y = np.cos(particles[:,4])
        pf_yaw[k] = np.arctan2(np.mean(sin_y), np.mean(cos_y))
        pf_bgrav[k] = np.mean(particles[:,5])

# For the first step (k=0) already set; but some steps where measurement happened we stored pf_x[k].
# For any zero entries left (if any) fill by nearest nonzero (simple fallback)
for arr in [pf_x, pf_y, pf_vx, pf_vy, pf_yaw, pf_bgrav]:
    mask = (arr == 0)
    if np.any(mask):
        # fill with INS where zeros (simple)
        arr[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], arr[~mask]) if np.any(~mask) else 0.0

# --- Evaluate errors ---
ins_err = np.sqrt((ins_x - true_x)**2 + (ins_y - true_y)**2)
pf_err = np.sqrt((pf_x - true_x)**2 + (pf_y - true_y)**2)

# --- Plots ---
plt.figure(figsize=(8, 6))
plt.plot(true_x, true_y, label='True trajectory')
plt.plot(ins_x, ins_y, label='INS dead-reckoning')
plt.plot(pf_x, pf_y, label='PF-corrected (Particle Filter)')
plt.legend()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectories (Particle Filter)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(times, ins_err, label='INS position error (m)')
plt.plot(times, pf_err, label='PF corrected error (m)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position error (m)')
plt.title('Position error vs time (Particle Filter)')
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, (g_map - g0) * 1e5, levels=40)
plt.colorbar(label='Gravity anomaly (1e-5 m/s^2)')
plt.plot(true_x, true_y, label='True')
plt.plot(ins_x, ins_y, label='INS')
plt.plot(pf_x, pf_y, label='PF corrected')
plt.legend()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Gravity anomaly map (overlay trajectories) - PF')
plt.show()

rmse_ins = np.sqrt(np.mean((ins_x - true_x)**2 + (ins_y - true_y)**2))
rmse_pf = np.sqrt(np.mean((pf_x - true_x)**2 + (pf_y - true_y)**2))
print(f"RMSE INS: {rmse_ins:.2f} m, RMSE PF-corrected: {rmse_pf:.2f} m")
