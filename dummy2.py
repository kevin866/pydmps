import numpy as np
import matplotlib.pyplot as plt
import pydmps
import pydmps.dmp_discrete

# Define sine and cosine trajectory
t = np.linspace(0, 2 * np.pi, 100)
y_des = np.array([np.sin(t), np.cos(t)])
print(y_des.shape)
# Initialize DMP
dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=2, n_bfs=500, ay=np.ones(2) * 10.0)

# Train DMP on the desired trajectory
dmp.imitate_path(y_des=y_des, plot=False)

# Modify initial conditions
dmp.y0 = np.array([0.5, 0.5])  # New starting point

# Generate trajectory
y_track, dy_track, ddy_track = dmp.rollout()
print(y_track.shape)
# Plot the trajectory
plt.figure(figsize=(6, 6))
plt.plot(y_track[:, 0], y_track[:, 1], "b", lw=2, label="Generated Trajectory")
plt.plot(y_des[:, 0], y_des[:, 1], "r--", lw=2, label="Demonstrated Trajectory")
plt.title("DMP with Modified Initial Conditions")
plt.legend()
plt.axis("equal")
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.show()
