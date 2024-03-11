import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})

matrix_sizes = np.array([100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000])
runtime = np.array([
0.836,
0.845,
0.853,
0.914,
1.025,
1.261,
1.464,
1.566,
1.803
])

runtime2 = np.array([
0.843,
0.846,
0.845,
0.904,
1.001,
1.261,
1.442,
1.559,
1.797
])

runtime3 = np.array([
0.832,
0.837,
0.857,
0.904,
1.014,
1.665,
1.467,
1.569,
1.690
])


avg = (runtime + runtime2 + runtime3)/3

fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xscale('log') # base=math.sqrt(2)
ax.set_yscale('log')
ax.set_xticks(matrix_sizes)
ax.set_xticklabels(['0.1', '0.5', '1', '5', '10', '20', '30', '40', '50'], rotation=45)

ax.plot(matrix_sizes, runtime, "--",linewidth=0.9, color='green')
ax.plot(matrix_sizes, runtime2, "--",linewidth=0.9, color='red')
ax.plot(matrix_sizes, runtime3, "--",linewidth=0.9, color='purple')

ax.plot(matrix_sizes, avg, 'o-', linewidth=3)


for i, proc in enumerate(runtime):
    plt.annotate('%0.3f' % proc, xy=(matrix_sizes[i], avg[i]), xytext=(-30, 10), textcoords='offset points', color='tab:blue')
ax.set_ylim([0.8, 1.85])
ax.set_xlim([50, 100000])

ax.set_title('Average Runtimes for Single-GPU CG')
ax.set_xlabel('Matrix sizes (x $10^3$)')
ax.set_ylabel('Walltime (s)', color="tab:blue")


plt.tight_layout()
plt.savefig('plot_single_gpu_runtime.pdf', format='pdf')
plt.show()