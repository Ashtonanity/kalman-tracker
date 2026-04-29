import numpy as np
import matplotlib.pyplot as plt

# ── TRUE TRAJECTORY ──────────────────────────────────────────
dt = 0.1
t  = np.arange(0, 20, dt)

true_x = t * 10
true_y = np.sin(t * 0.5) * 100

# ── NOISY GPS MEASUREMENTS ───────────────────────────────────
np.random.seed(42)
gps_noise = 15
meas_x = true_x + np.random.normal(0, gps_noise, len(t))
meas_y = true_y + np.random.normal(0, gps_noise, len(t))

# ── KALMAN FILTER SETUP ──────────────────────────────────────
# State vector: [x, y, vx, vy] — position AND velocity
# This is the key insight: we track velocity too, not just position

# State transition matrix — physics model (constant velocity)
F = np.array([[1, 0, dt, 0],
              [0, 1, 0, dt],
              [0, 0, 1,  0],
              [0, 0, 0,  1]])

# Measurement matrix — we only observe x and y, not velocity
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])

# Process noise — how much we trust our physics model
Q = np.eye(4) * 1.0

# Measurement noise — how much we trust the GPS (matches gps_noise)
R = np.eye(2) * (gps_noise ** 2)

# Initial state and uncertainty
x_est = np.array([meas_x[0], meas_y[0], 0, 0])
P     = np.eye(4) * 500  # high initial uncertainty

# ── FILTER LOOP ──────────────────────────────────────────────
filtered_x = []
filtered_y = []

for i in range(len(t)):
    # PREDICT
    x_pred = F @ x_est
    P_pred = F @ P @ F.T + Q

    # UPDATE
    z   = np.array([meas_x[i], meas_y[i]])         # measurement
    y   = z - H @ x_pred                            # innovation
    S   = H @ P_pred @ H.T + R                      # innovation covariance
    K   = P_pred @ H.T @ np.linalg.inv(S)           # Kalman gain
    x_est = x_pred + K @ y                          # updated state
    P   = (np.eye(4) - K @ H) @ P_pred              # updated uncertainty

    filtered_x.append(x_est[0])
    filtered_y.append(x_est[1])

# ── PLOT ─────────────────────────────────────────────────────
plt.figure(figsize=(14, 6))
plt.plot(true_x, true_y,     'g-',  label='True Path',        linewidth=2)
plt.plot(meas_x, meas_y,     'r.',  label='Noisy GPS',        alpha=0.4)
plt.plot(filtered_x, filtered_y, 'b-', label='Kalman Estimate', linewidth=2)
plt.legend(fontsize=12)
plt.title('Kalman Filter — Aircraft Position Tracking', fontsize=14)
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.tight_layout()
plt.savefig('kalman_output.png', dpi=150)
plt.show()
print("Done — saved to kalman_output.png")
