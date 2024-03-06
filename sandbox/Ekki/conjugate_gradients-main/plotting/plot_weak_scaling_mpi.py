import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['text.usetex'] = True

matrix_sizes = np.array([1000, 1414, 2000, 2828, 4000, 5656, 8000, 11314])
runtime = np.array([1.33274, 1.3744, 1.40032, 1.41522, 1.44147, 1.63674, 1.72475, 1.85462])
efficiency = runtime[0] / runtime

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(matrix_sizes, runtime, 'o-')
ax.set_xscale('log', base=math.sqrt(2))
ax.set_xticks(matrix_sizes)
ax.set_xticklabels(['1000 (1)', '1414 (2)', '2000 (4)', '2828 (8)', '4000 (16)', '5656 (32)', '8000 (64)', '11314 (128)'])
for i, proc in enumerate(runtime):
    plt.annotate('%0.3f' % proc, xy=(matrix_sizes[i], runtime[i]), xytext=(-9, 6), textcoords='offset points')
ax.set_ylim([0, 16])
ax.set_title('Runtimes for MPI CG (Weak Scaling)')
ax.set_xlabel('Matrix sizes (Number of processes)')
ax.set_ylabel('Walltime (s)', color="tab:blue")

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Efficiency', color="green")  # we already handled the x-label with ax1
ax2.plot(matrix_sizes, efficiency, color="green")
for i, proc in enumerate(efficiency):
    plt.annotate('%0.2f' % proc, xy=(matrix_sizes[i], efficiency[i]), xytext=(-9, 3), textcoords='offset points')
ax2.set_ylim([0, 1.6])


plt.savefig('plot_weak_scaling_mpi.pdf', format='pdf')
plt.show()