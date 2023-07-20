import numpy as np
import matplotlib.pyplot as plt

# Finds Kalman Filter and shows the state estimates

x = np.array([0, 0]).reshape(-1, 1)  # state vector
xh = np.array([0, 0]).reshape(-1, 1)  # Kalman estimate vector of state
Npoints = 200  # number of time steps
y = np.zeros((Npoints, 1))
s = np.zeros(Npoints)
x1 = np.zeros(Npoints)
x2 = np.zeros(Npoints)
xh1 = np.zeros(Npoints)
xh2 = np.zeros(Npoints)
t = np.arange(1, Npoints + 1)

# Define the system
F = np.array([[0, 1], [-0.8, 1.5]])  # system matrix
D = np.array([0, 1]).reshape(-1, 1)  # system process noise vector
H = np.array([[1, 1]])  # observation matrix
Q = D @ D.T  # Process noise covariance is unity

R = 5  # Measurement noise variance - scalar

# Initialize the covariance matrix
P = np.eye(2) * 0.1  # Covariance matrix for initial state error

# Generate Random Noise Variance =1
zeta = np.random.randn(Npoints)

# Re-seed noise generator
np.random.seed(2)

# Additive uncorrelated white noise
rv = np.sqrt(10) * np.random.randn(Npoints)

# Loop through and perform the Kalman filter equations recursively
for k in range(Npoints):
    # State equations of system
    x = F @ x + D * zeta[k]
    y[k] = H @ x
    s[k] = y[k] + rv[k]

    # Store states for plotting later
    x1[k] = x[0]
    x2[k] = x[1]

    # Update Kalman gain vector: one step ahead version
    K = F @ P @ H.T / (H @ P @ H.T + R)

    # Update the covariance from the Riccati equation
    P = F @ P @ F.T + Q - K @ H @ P @ F.T

    # Kalman Filter
    xh = F @ xh + K * (s[k] - H @ xh)

    # Store estimated states for plotting
    xh1[k] = xh[0]
    xh2[k] = xh[1]

# Plot results
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(t, x1, '-r', t, xh1, '-.b')
plt.legend(['State 1', 'State 1 estimate'], loc='upper left')
plt.title('State 1 and KF Estimate')
plt.xlabel('Time(samples)')

plt.subplot(2, 1, 2)
plt.plot(t, x2, '-r', t, xh2, '-.b')
plt.legend(['State 2', 'State 2 estimate'], loc='upper left')
plt.title('State 2 and KF Estimate')
plt.xlabel('Time(samples)')

plt.tight_layout()
plt.show()
