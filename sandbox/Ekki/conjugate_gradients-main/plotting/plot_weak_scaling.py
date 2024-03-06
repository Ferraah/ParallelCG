import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['text.usetex'] = True

matrix_sizes = np.array([10000, 14142, 20000, 24492, 28288])
runtime = np.array([11.4240000000, 11.9455120000, 12.2781300000, 12.2975240000, 12.4058600000])
efficiency = np.array([1, 0.96, 0.93, 0.93, 0.92])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(matrix_sizes, runtime, 'o-')
ax.set_xscale('log', base=math.sqrt(2))
ax.set_xticks(matrix_sizes)
ax.set_xticklabels(['10000 (1)', '14142 (2)', '20000 (4)', '24492 (6)', '28288 (8)'])
for i, proc in enumerate(runtime):
    plt.annotate('%0.3f' % proc, xy=(matrix_sizes[i], runtime[i]), xytext=(-9, 6), textcoords='offset points')
ax.set_ylim([0, 16])
ax.set_title('Runtimes for powermethod (Weak Scaling)')
ax.set_xlabel('Matrix sizes (Number of processes)')
ax.set_ylabel('Walltime (s)', color="tab:blue")

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Efficiency', color="green")  # we already handled the x-label with ax1
ax2.plot(matrix_sizes, efficiency, color="green")
for i, proc in enumerate(efficiency):
    plt.annotate('%0.2f' % proc, xy=(matrix_sizes[i], efficiency[i]), xytext=(-9, 3), textcoords='offset points')
ax2.set_ylim([0, 1.6])




plt.savefig('plot_weak_scaling.pdf', format='pdf')
plt.show()