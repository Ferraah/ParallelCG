import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})

matrix_sizes = np.array([5000, 10000, 20000])
runtime = np.array([
0.924,
0.710,
1.166,
])
efficiency = runtime[0] / runtime

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(matrix_sizes, runtime, 'o-')
ax.set_xscale('log', base=math.sqrt(2))
ax.set_xticks(matrix_sizes)
ax.set_xticklabels(['5 (1)', '10 (4)', '20 (16)',])
for i, proc in enumerate(runtime):
    plt.annotate('%0.3f' % proc, xy=(matrix_sizes[i], runtime[i]), xytext=(-9, 6), textcoords='offset points', color='tab:blue')
ax.set_ylim([0, 16])
ax.set_title('Runtimes for Multi-GPU CG  (Weak Scaling)')
ax.set_xlabel('Matrix sizes (x $10^3$) (Number of GPUs)')
ax.set_ylabel('Walltime (s)', color="tab:blue")

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Efficiency', color="green")  # we already handled the x-label with ax1
ax2.plot(matrix_sizes, efficiency, color="green")
for i, proc in enumerate(efficiency):
    plt.annotate('%0.2f' % proc, xy=(matrix_sizes[i], efficiency[i]), xytext=(-9, 3), textcoords='offset points', color='green')
ax2.set_ylim([0, 1.6])

plt.tight_layout()
plt.savefig('plot_weak_scaling_mpi.pdf', format='pdf')
plt.show()