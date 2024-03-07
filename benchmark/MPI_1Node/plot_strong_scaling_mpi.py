import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})

processes = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
runtime = np.array([145.168, 72.118, 36.1497, 18.1116, 9.20592, 5.09505, 2.65923, 1.45578, 1.35151])
efficiency = runtime[0] / (processes * runtime)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(processes, runtime, 'o-')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
ax.set_xticks(processes)
ax.set_xticklabels(['1', '2', '4', '8', '16', '32', '64', '128', '256'])
for i, proc in enumerate(runtime):
    plt.annotate('%0.3f' % proc, xy=(processes[i], runtime[i]), xytext=(7, 2), textcoords='offset points', color='tab:blue')
ax.set_title('Runtimes for MPI CG (Strong Scaling)')
ax.set_xlabel('Number of processes')
ax.set_ylabel('Walltime (s)', color="tab:blue")

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Efficiency', color="green")  # we already handled the x-label with ax1
ax2.plot(processes, efficiency, color="green")
for i, proc in enumerate(efficiency):
    plt.annotate('%0.3f' % proc, xy=(processes[i], efficiency[i]), xytext=(7, 2), textcoords='offset points', color='green')
ax2.set_ylim([0, 1.3])

plt.savefig('plot_strong_scaling_mpi.pdf', format='pdf')
plt.show()