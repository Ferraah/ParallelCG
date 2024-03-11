
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LogNorm



plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})


x = np.array([  # Sample data (replace with your actual data)
    [2, 99, 88, 84, 168, 35, 0, 0, 0, 0],
    [7, 60, 94, 83, 94, 50, 57, 141, 0, 0],
    [40, 65, 84, 112, 145, 36, 94, 129, 165, 0],
    [227, 169, 174, 246, 178, 75, 114, 180, 206, 819],
    [778, 403, 550, 598, 419, 273, 169, 295, 294, 693],
    [3316, 1511, 1483, 1236, 865, 423, 203, 378, 528, 925],
    [7702, 3403, 2924, 2744, 1688, 1000, 434, 693, 588, 1443],
    [15397, 6095, 6658, 4581, 3130, 1258, 909, 743, 888, 2030],
    [22750, 9463, 14426, 5623, 4620, 2360, 1149, 1008, 1155, 3662]
])
procs = np.array([1, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
dims = np.array([100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000])
fig, ax = plt.subplots(figsize=(12, 12))

# Assuming x[i][j] represents the time taken for process i and dimension j

x = x/1000
# Enforce non-zero values (adjust threshold if needed)
x = np.where(x == 0, 1e-6, x)  # Replace zeros with a small positive value

y = [x[j][i] for j in range(len(dims)) for i in range(len(procs))]  # Extract time taken for each combination
Z = np.array(x)  # Assuming y is already a properly shaped numpy array

procs = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
dims = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

for (i, j), z in np.ndenumerate(x):
  if x[i][j] == 1e-6:
    ax.text(procs[j], dims[i], "ND", ha='center', va='center', fontsize=15, bbox=dict(boxstyle='square', facecolor='white', edgecolor='0.3')) 
  else:
    ax.text(procs[j], dims[i], f"{z:.3f}", ha='center', va='center', fontsize=15, bbox=dict(boxstyle='square', facecolor='white', edgecolor='0.3'))

pcolor = ax.pcolormesh(procs, dims, x, cmap='viridis', norm=LogNorm(vmin=0.04, vmax=25))

ax.set_xlabel('MPI Processes')
ax.set_ylabel('Matrix dimension (x$10^3$)')
ax.set_title('Timing of IO operations for increasing size and active processes')
fig.colorbar(pcolor, label="Time (Log) (s)")
# Add colorbar for legend

# Set viewing angles for better visualization

# Show the plot

ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
ax.set_yticks([1, 2, 3, 4, 5, 6, 7, 8, 9])
ax.set_xticklabels(['1', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])
ax.set_yticklabels(['0.1', '0.5', '1', '5', '10', '20', '30', '40', '50'])


# Add colorbar for the colormap
plt.tight_layout()
plt.show()

plt.savefig("bench_io.pdf", format='pdf')