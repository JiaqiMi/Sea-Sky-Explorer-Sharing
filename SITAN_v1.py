# Python SITAN demo: 2D planar inertial navigation + scalar gravity matching EKF (SITAN-like)
# This code simulates a vehicle moving in a horizontal plane, creates a scalar gravity map with
# small anomalies, simulates IMU (accelerometer+gyro) and a scalar gravimeter (measuring gravity magnitude),
# runs an INS dead-reckoning solution and an error-state EKF that fuses scalar gravity measurements
# to correct position (a simplified SITAN demo).
#
# Requirements: numpy, scipy, matplotlib, pandas (all are usually available in the execution env).
#
# Run this cell to simulate and produce plots: true trajectory, INS-only (dead reckoning), and SITAN-corrected.

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

np.random.seed(2)

# --- Simulation parameters ---
dt = 0.1                     # IMU timestep (s)
T = 200.0                    # total time (s)
steps = int(T / dt)
times = np.arange(0, steps) * dt

# Map / area
xmin, xmax, ymin, ymax = -250.0, 250.0, -250.0, 250.0
grid_res = 2.0               # meters
xs = np.arange(xmin, xmax + grid_res, grid_res)
ys = np.arange(ymin, ymax + grid_res, grid_res)
XX, YY = np.meshgrid(xs, ys, indexing='xy')

# Base gravity and anomalies (units: m/s^2). We'll create anomalies on the order of e-5 (a few mGal).
g0 = 9.80665  # nominal gravity in m/s^2

# Create several Gaussian bumps as anomalies (positive or negative)
anomalies = np.zeros_like(XX)
bumps = [
    ( 60,  80,  0.00002, 40.0),
    (-90, -50, -0.000015, 60.0),
    (  0,   0,  0.00003, 30.0),
    (120, -120, 0.00001, 50.0)
]
for cx, cy, amp, sigma in bumps:
    anomalies += amp * np.exp(-(((XX - cx)**2 + (YY - cy)**2) / (2 * sigma**2)))

# Smooth anomalies a little to avoid grid artifacts
anomalies = gaussian_filter(anomalies, sigma=1.0)

# Scalar gravity map = g0 + anomalies
g_map = g0 + anomalies

# Pre-compute gradient (partial derivatives wrt x and y) using central differences via numpy.gradient
# Note: gradient returns derivatives w.r.t axis 0 (y) and axis 1 (x) when indexing='xy' and meshgrid as above.
dy, dx = np.gradient(g_map, grid_res, grid_res)  # dy = ∂g/∂y, dx = ∂g/∂x
# We'll use gradient vector [dg/dx, dg/dy].
# Create interpolators for g and gradients
g_interp = RegularGridInterpolator((xs, ys), g_map.T, bounds_error=False, fill_value=None)
dgdx_interp = RegularGridInterpolator((xs, ys), dx.T, bounds_error=False, fill_value=0.0)
dgdy_interp = RegularGridInterpolator((xs, ys), dy.T, bounds_error=False, fill_value=0.0)

# --- True trajectory (2D) ---
# We'll create a smooth path that traverses the anomalies: a sinusoidal track.
speed = 2.0  # m/s approximately constant speed
total_length = speed * T
# Parametric path: circle-like + sine modulation
theta_path = np.linspace(0, 4*np.pi, steps)  # two revolutions
r = 120 + 60 * np.sin(0.5 * theta_path)
true_x = r * np.cos(theta_path)
true_y = r * np.sin(theta_path) + 30 * np.sin(0.2 * theta_path)
# Compute true velocities by finite difference
true_vx = np.gradient(true_x, dt)
true_vy = np.gradient(true_y, dt)
# Compute true yaw (heading) from velocity direction
true_yaw = np.arctan2(true_vy, true_vx)
# Smooth yaw a bit
true_yaw = gaussian_filter(true_yaw, sigma=2)

# True accelerations (in nav frame)
true_ax = np.gradient(true_vx, dt)
true_ay = np.gradient(true_vy, dt)

# --- IMU simulation (accelerometer and gyro) ---
# We'll assume the vehicle has a body frame rotated by yaw around z.
# Specific force in nav frame = acceleration (horizontal) (ax, ay) and vertical acceleration = 0.
# Gravity vector in nav frame is [0, 0, - (g0 + anomaly)], where anomaly depends on x,y.
# Accelerometer measures body-frame specific force: f_b = C_nb^T * (a_nav - g_nav) + accel_bias + noise
# For planar motion, a_nav = [ax, ay, 0], g_nav = [0, 0, -(g0 + anom)], so a_nav - g_nav = [ax, ay, g0+anom].
# Accelerometer senses the "upward" specific force component as positive z in body frame.
accel_noise_std = 0.02  # m/s^2 (accelerometer noise)
gyro_noise_std = 0.001  # rad/s
accel_bias_true = np.array([0.02, -0.01, 0.0])  # small bias in body frame (x,y,z)
gyro_bias_true = 0.002  # rad/s yaw bias

# For simplicity, we simulate only x,y accel components in body frame and treat vertical as a scalar
# But we'll produce 3-axis accel for mechanization consistency.

accel_meas = np.zeros((steps, 3))
gyro_meas = np.zeros(steps)
gravimeter_meas = np.zeros(steps)

for k in range(steps):
    xk = true_x[k]
    yk = true_y[k]
    # gravity magnitude at true position
    g_at = g_interp((xk, yk))  # m/s^2
    # specific force in nav frame: a_nav - g_nav (g_nav = [0,0,-g_at])
    a_nav = np.array([true_ax[k], true_ay[k], 0.0])
    specific_force_nav = a_nav - np.array([0, 0, -g_at])  # = [ax, ay, g_at]
    # rotate to body frame via yaw
    yaw = true_yaw[k]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw),  np.cos(yaw), 0],
                     [0,             0,          1]])
    f_b_true = C_nb.T.dot(specific_force_nav)  # 3-vector
    # add bias and noise
    accel_meas[k] = f_b_true + accel_bias_true + np.random.randn(3) * accel_noise_std
    # gyro measures yaw rate (approx from derivative of yaw)
    if k == 0:
        yaw_rate = 0.0
    else:
        yaw_rate = (true_yaw[k] - true_yaw[k-1]) / dt
    gyro_meas[k] = yaw_rate + gyro_bias_true + np.random.randn() * gyro_noise_std
    # gravimeter measures scalar gravity magnitude at vehicle location (with small bias and noise)
    grav_bias_true = 0.0  # assume zero true bias for gravimeter; we may estimate a small sensor bias
    grav_noise_std = 5e-6  # m/s^2 (~0.5 mGal)
    gravimeter_meas[k] = g_at + grav_bias_true + np.random.randn() * grav_noise_std

# --- INS dead-reckoning (simple mechanization using measured accel and gyro) ---
# Initial conditions: use true initial position but INS does not know biases -> we add biases to simulate drift.
ins_x = np.zeros(steps)
ins_y = np.zeros(steps)
ins_vx = np.zeros(steps)
ins_vy = np.zeros(steps)
ins_yaw = np.zeros(steps)

# Start at a perturbed initial position
ins_x[0] = true_x[0] + 5.0   # 5 m initial horizontal error
ins_y[0] = true_y[0] - 3.0   # 3 m initial error
ins_vx[0] = true_vx[0]
ins_vy[0] = true_vy[0]
ins_yaw[0] = true_yaw[0] + 0.05  # small heading error

for k in range(1, steps):
    # integrate gyro to update yaw
    ins_yaw[k] = ins_yaw[k-1] + (gyro_meas[k-1]) * dt  # using measured gyro (includes bias)
    # get acceleration in body measured, remove no bias knowledge -> INS uses raw accel_meas
    f_b = accel_meas[k-1]
    # convert to nav frame
    yaw = ins_yaw[k-1]
    C_nb = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                     [np.sin(yaw),  np.cos(yaw), 0],
                     [0,             0,          1]])
    # specific force in nav: R * f_b
    specific_force_nav = C_nb.dot(f_b)
    # subtract gravity nominal g0 in z to get horizontal acceleration approx (we don't have anomaly in INS)
    a_nav = specific_force_nav.copy()
    # Horizontal accelerations (assume third component is vertical, ignore)
    ax = a_nav[0]
    ay = a_nav[1]
    # integrate velocity and position
    ins_vx[k] = ins_vx[k-1] + ax * dt
    ins_vy[k] = ins_vy[k-1] + ay * dt
    ins_x[k] = ins_x[k-1] + ins_vx[k-1] * dt + 0.5 * ax * dt**2
    ins_y[k] = ins_y[k-1] + ins_vy[k-1] * dt + 0.5 * ay * dt**2

# --- SITAN-like error-state EKF using scalar gravity measurements ---
# State: [dx, dy, dvx, dvy, dtheta, bax, bay, bgrav] where bgrav is gravimeter bias (scalar)
nx = 8
x_est = np.zeros((steps, nx))
P = np.zeros((steps, nx, nx))

# Initialize error-state: start with difference between INS and true
x_est[0, :] = np.array([ins_x[0] - true_x[0],
                        ins_y[0] - true_y[0],
                        ins_vx[0] - true_vx[0],
                        ins_vy[0] - true_vy[0],
                        ins_yaw[0] - true_yaw[0],
                        0.0, 0.0, 0.0])  # initial bias errors (we'll let filter estimate some)
P0 = np.diag([25.0, 25.0, 1.0, 1.0, 0.1**2, 0.05**2, 0.05**2, (1e-5)**2])  # covariances
P[0] = P0.copy()

# Process noise Q (for error-state)
# noise for position-velocity small, orientation and biases random walk
q_pos = 0.01
q_vel = 0.1
q_theta = 1e-6
q_ba = 1e-6
q_bgrav = (1e-7)**2
Q = np.diag([q_pos, q_pos, q_vel, q_vel, q_theta, q_ba, q_ba, q_bgrav])

# Measurement noise R for gravimeter (variance)
R_grav = grav_noise_std**2 + (1e-6)**2  # include some map uncertainty

# We'll perform filter in discrete-time with linearized F computed at each step
# For convenience, we will maintain a "corrected" INS state which applies the estimated error-state to INS solution
corr_x = np.zeros(steps)
corr_y = np.zeros(steps)
corr_vx = np.zeros(steps)
corr_vy = np.zeros(steps)
corr_yaw = np.zeros(steps)

# initialize corrected state with INS initial minus estimated error
corr_x[0] = ins_x[0] - x_est[0, 0]
corr_y[0] = ins_y[0] - x_est[0, 1]
corr_vx[0] = ins_vx[0] - x_est[0, 2]
corr_vy[0] = ins_vy[0] - x_est[0, 3]
corr_yaw[0] = ins_y[0] - x_est[0, 4]

# measurement rate: assume gravimeter at 1 Hz
meas_interval_steps = int(1.0 / dt)

for k in range(1, steps):
    # --- Predict step: error-state propagation (linearized) ---
    # We need approximate measured body acceleration at previous step used by INS for propagation
    # Use accel_meas from k-1 (contains true f_b + bias + noise)
    f_b = accel_meas[k-1]  # 3-vector measured
    # estimated yaw from corrected state
    yaw_est = corr_yaw[k-1]
    C_nb_est = np.array([[np.cos(yaw_est), -np.sin(yaw_est), 0],
                         [np.sin(yaw_est),  np.cos(yaw_est), 0],
                         [0,                 0,               1]])
    # Build continuous-time linearized F and discretize by simple Euler (F_d = I + F*dt)
    F = np.zeros((nx, nx))
    # dx/dt = dv
    F[0, 2] = 1.0
    F[1, 3] = 1.0
    # dv/dt depends on orientation error and accel bias error:
    # dv_dot ≈ C_nb * ( - [f_b]_x * delta_theta - delta_ba )
    # so partial d(dv)/dtheta = C_nb * ( -[f_b]_x )
    # and partial d(dv)/d(bax,bay) = -C_nb[:,0:2] (only x,y biases)
    fb_x = np.array([[0, -f_b[2], f_b[1]],
                     [f_b[2], 0, -f_b[0]],
                     [-f_b[1], f_b[0], 0]])
    # take top-left 2x3 of C_nb * (-fb_x) for dvx/dtheta and dvy/dtheta (only z-rotation matters)
    temp = C_nb_est.dot(-fb_x)  # 3x3
    # only z-rotation (heading) influences horizontal acceleration in planar motion; map to scalar delta_theta around z
    # So use columns corresponding to rotation about z basis vector e3 -> column index 2
    F[2, 4] = temp[0, 2]  # dvx / dtheta
    F[3, 4] = temp[1, 2]  # dvy / dtheta
    # dv / d bax,bay (accelerometer biases) : dv = - C_nb * delta_ba
    F[2, 5] = -C_nb_est[0, 0]  # dvx/d bax (approx)
    F[2, 6] = -C_nb_est[0, 1]  # dvx/d bay
    F[3, 5] = -C_nb_est[1, 0]  # dvy/d bax
    F[3, 6] = -C_nb_est[1, 1]  # dvy/d bay
    # dtheta/d (gyro bias) : orientation error integrates gyro bias error negatively
    F[4, 4] = 0.0  # small-angle; we'll add bias coupling below
    # dtheta/d (gyro bias) negative influence: d(delta_theta)/dt = - delta_bg (bg is gyro bias), we represent bg in state? not here explicitly
    # But we did not include gyro bias state; orientation error driven by gyro bias is treated as process noise
    # biases random walk
    # bax_dot = noise -> leave F rows zeros for biases (random walk)
    # gravimeter bias bgrav random walk: leave zero

    # Discrete-time transition
    Fd = np.eye(nx) + F * dt

    # Propagate covariance
    P_pred = Fd.dot(P[k-1]).dot(Fd.T) + Q * dt

    # Predict state: in error-state EKF the nominal INS state is integrated outside; error-state x evolves approx as:
    # x = Fd * x_prev (neglecting small driving terms)
    x_pred = Fd.dot(x_est[k-1])

    # --- Optionally, also propagate corrected INS nominal using INS mechanization (we already have ins_x etc) ---
    # For corrected nominal state, we apply no propagation here; we will update when measurement arrives.
    # But to keep a corrected nominal trajectory, we update corr_* with INS dead-reckoned motion (already computed)
    corr_x[k] = ins_x[k] - x_pred[0]
    corr_y[k] = ins_y[k] - x_pred[1]
    corr_vx[k] = ins_vx[k] - x_pred[2]
    corr_vy[k] = ins_vy[k] - x_pred[3]
    corr_yaw[k] = ins_yaw[k] - x_pred[4]

    # Save predicted state and P temporarily
    x_est[k] = x_pred
    P[k] = P_pred

    # --- Measurement update when gravimeter sample is available ---
    if (k % meas_interval_steps) == 0:
        # measurement z at time k (gravimeter_meas)
        z = gravimeter_meas[k]

        # compute map predicted gravity at nominal corrected INS position (corr_x, corr_y)
        pos_nom = np.array([corr_x[k], corr_y[k]])
        g_pred = g_interp((pos_nom[0], pos_nom[1]))
        # innovation
        y = z - g_pred

        # compute gradient at nominal position
        dgdx = dgdx_interp((pos_nom[0], pos_nom[1]))
        dgdy = dgdy_interp((pos_nom[0], pos_nom[1]))
        grad = np.array([dgdx, dgdy])  # (2,)

        # Build H: measurement depends on position error (dx,dy) and gravimeter bias bgrav
        H = np.zeros((1, nx))
        H[0, 0] = grad[0]  # ∂g/∂x * dx
        H[0, 1] = grad[1]  # ∂g/∂y * dy
        # rest zeros except grav bias
        H[0, 7] = 1.0      # measurement includes grav bias additively (we included bgrav in state)

        # Innovation covariance and Kalman gain
        S = H.dot(P_pred).dot(H.T) + R_grav
        K = P_pred.dot(H.T).dot(np.linalg.inv(S))
        # update error-state
        dx = (K.dot(np.array([[y]]))).reshape(-1)
        x_upd = x_pred + dx
        # Joseph form covariance update
        I = np.eye(nx)
        P_upd = (I - K.dot(H)).dot(P_pred).dot((I - K.dot(H)).T) + K.dot(R_grav).dot(K.T)

        # apply correction to nominal INS state (inject error state)
        # corrected nominal = nominal - estimated error (error-state definition depends; here x stores INS - true so -x corrects)
        corr_x[k] = corr_x[k] - x_upd[0]
        corr_y[k] = corr_y[k] - x_upd[1]
        corr_vx[k] = corr_vx[k] - x_upd[2]
        corr_vy[k] = corr_vy[k] - x_upd[3]
        corr_yaw[k] = corr_yaw[k] - x_upd[4]
        # Reset error-state after injection to zero (typical error-state approach: set state to zero after applying)
        x_est[k] = np.zeros(nx)
        P[k] = P_upd

# --- Evaluate results: compute position errors ---
ins_err = np.sqrt((ins_x - true_x)**2 + (ins_y - true_y)**2)
sitan_err = np.sqrt((corr_x - true_x)**2 + (corr_y - true_y)**2)

# --- Plot trajectories: true, INS dead-reckoning, SITAN-corrected ---
plt.figure(figsize=(8, 6))
plt.plot(true_x, true_y, label='True trajectory')
plt.plot(ins_x, ins_y, label='INS dead-reckoning')
plt.plot(corr_x, corr_y, label='SITAN-corrected (EKF)')
plt.legend()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Trajectories')
plt.grid(True)
plt.show()

# Plot position error over time
plt.figure(figsize=(8, 4))
plt.plot(times, ins_err, label='INS position error (m)')
plt.plot(times, sitan_err, label='SITAN corrected error (m)')
plt.legend()
plt.xlabel('Time (s)')
plt.ylabel('Position error (m)')
plt.title('Position error vs time')
plt.grid(True)
plt.show()

# Plot gravity map and trajectories overlay for context
plt.figure(figsize=(8, 6))
plt.contourf(XX, YY, (g_map - g0) * 1e5, levels=40)  # show anomaly in units of 1e-5 m/s^2 (roughly mGal)
plt.colorbar(label='Gravity anomaly (1e-5 m/s^2)')
plt.plot(true_x, true_y, label='True')
plt.plot(ins_x, ins_y, label='INS')
plt.plot(corr_x, corr_y, label='SITAN corrected')
plt.legend()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Gravity anomaly map (overlay trajectories)')
plt.show()

# Print simple RMSE numbers
rmse_ins = np.sqrt(np.mean((ins_x - true_x)**2 + (ins_y - true_y)**2))
rmse_sitan = np.sqrt(np.mean((corr_x - true_x)**2 + (corr_y - true_y)**2))
print(f"RMSE INS: {rmse_ins:.2f} m, RMSE SITAN-corrected: {rmse_sitan:.2f} m")
