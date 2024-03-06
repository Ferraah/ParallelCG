import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True

processes = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])
runtime = np.array([37.2649, 18.9598, 9.95739, 8.88727, 8.48109, 8.71838, 10.8129, 11.981, 12.188])
efficiency = runtime[0] / (processes * runtime)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(processes, runtime, 'o-')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
ax.set_xticks(processes)
ax.set_xticklabels(['1', '2', '4', '6', '8', '10', '12', '14', '16'])
for i, proc in enumerate(runtime):
    plt.annotate('%0.3f' % proc, xy=(processes[i], runtime[i]), xytext=(7, 2), textcoords='offset points')
ax.set_title('Runtimes for OpenMP CG (Strong Scaling)')
ax.set_xlabel('Number of threads')
ax.set_ylabel('Walltime (s)', color="tab:blue")

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Efficiency', color="green")  # we already handled the x-label with ax1
ax2.plot(processes, efficiency, color="green")
for i, proc in enumerate(efficiency):
    plt.annotate('%0.3f' % proc, xy=(processes[i], efficiency[i]), xytext=(7, 2), textcoords='offset points')
ax2.set_ylim([0, 1.3])

plt.savefig('plot_strong_scaling_omp.pdf', format='pdf')
plt.show()