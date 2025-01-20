"""
Copyright (C) 2016 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""


import numpy as np
import matplotlib.pyplot as plt

import pydmps
import pydmps.dmp_discrete
y_tracks = []
ic = np.array([-1, 0.5, 1])
goal = np.array([2.0])
# Generated by ChatGPT
t = np.linspace(0, 2 * np.pi, 100)
# sin function
y_des = np.array([np.sin(t)])

dmp = pydmps.dmp_discrete.DMPs_discrete(n_dmps=1, n_bfs=500, ay=np.ones(1) * 10.0)
print(type(np.ones(2) * 10.0))
dmp.imitate_path(y_des=y_des, plot=False)

for i in range(len(ic)):
    dmp.y0 = np.array([ic[i]])  # Generated by ChatGPT
    dmp.goal = goal # Generated by ChatGPT
    y_track = []
    dy_track = []
    ddy_track = []
    y_track, dy_track, ddy_track = dmp.rollout()
    y_tracks.append(y_track)

print(len(y_tracks))
plt.figure(1, figsize=(6, 6))
plt.plot(y_tracks[0], "b", lw=2)
plt.plot(y_tracks[1], "r", lw=2)
plt.plot(y_tracks[2], "g", lw=2)

plt.legend(["ic = -1","ic = 0.5","ic = 1"])
plt.title("DMP system with sine input function")

plt.show()



