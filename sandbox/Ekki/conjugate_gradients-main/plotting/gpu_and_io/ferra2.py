import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle  # Import Rectangle patch
from matplotlib.colors import LogNorm

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 20})

times = np.array([31.52333, 17.09744, 9.94449, 6.86093, 7.38143, 8.50476, 19.21317, 42.20377, 216.527])
matrix_sizes = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])


'1 (256)','2 (128)','4 (64)','8 (32)','16 (16)','32 (8)','64 (4)','128 (2)','256 (1)'


x = np.linspace(1,9,9)
y = np.cumsum(np.random.randn(50))+6

fig, ax = plt.subplots(figsize=(10,6))

extent = [x[0]-(x[1]-x[0])/2., x[-1]+(x[1]-x[0])/2., 0, np.max(times)]  # Adjust upper limit for log scale
im = ax.imshow(times.reshape(1,len(times)), cmap="viridis", aspect="auto", extent=extent, norm=LogNorm(vmin=7))
ax.set_yticks([])
ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9], labels=['1 (256)','2 (128)','4 (64)','8 (32)','16 (16)','32 (8)','64 (4)','128 (2)','256 (1)'], rotation=45)
ax.set_xlim(extent[0], extent[1])
ax.set_xlabel("Processes (Threads)")  # Add y-axis label for clarity

ax.set_title("Time execution (with IO) of \n varying active MPI processes and OpenMP threads")

for i in range(len(matrix_sizes)):
    # Create a square patch with slightly smaller width/height for spacing
    x_center = i + 1
    y_center = 105
    width = 0.8  # Adjust width for desired size (less than 1 for spacing)
    height = 12 # Adjust height for desired size (less than 1 for spacing)
    square = Rectangle(xy=(x_center - width / 2, y_center - height / 2),
                        width=width, height=height, color='white', edgecolor='black', linewidth=2)
    ax.add_patch(square)

    # Add text caption with execution time (adjust font size and position as needed)
    ax.text(x_center, y_center, f"{times[i]:.2f}", ha='center', va='center', fontsize=13, fontweight='bold')

cbar = fig.colorbar(im, ax=ax, label="Execution Time (Log) (s)")

# plt.tight_layout()
plt.tight_layout()
plt.savefig("risultato.png")
plt.show()