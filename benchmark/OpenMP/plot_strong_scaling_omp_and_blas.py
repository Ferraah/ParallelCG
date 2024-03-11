import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})

processes = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16])
runtime_omp = np.array([37.2649, 18.9598, 9.95739, 8.88727, 8.48109, 8.71838, 10.8129, 11.981, 12.188])
runtime_omp_blas = np.array([20.6965, 11.0413, 9.24838, 7.30625, 7.82275, 8.0043, 8.52555, 11.4798, 10.0142])
efficiency_omp = runtime_omp[0] / (processes * runtime_omp)
efficiency_omp_blas = runtime_omp_blas[0] / (processes * runtime_omp_blas)

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(processes, runtime_omp, 'o-', label='omp', color='tab:blue')
ax.plot(processes, runtime_omp_blas, 'o-', color='blue', label='omp+blas')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
ax.set_xticks(processes)
ax.set_xticklabels(['1', '2', '4', '6', '8', '10', '12', '14', '16'])
# ax.plot((), color='tab:blue', label='omp')
# ax.plot((), color='blue', label='omp+blas')
for i, proc in enumerate(runtime_omp[:5]):
    plt.annotate('%0.3f' % proc, xy=(processes[i], runtime_omp[i]), xytext=(7, 2), textcoords='offset points', color='tab:blue')
for i, proc in enumerate(runtime_omp_blas[:5]):
    plt.annotate('%0.3f' % proc, xy=(processes[i], runtime_omp_blas[i]), xytext=(7, 2), textcoords='offset points', color='blue')
ax.set_title('Runtimes for OpenMP and OpenBLAS CG (Strong Scaling)')
ax.set_xlabel('Number of threads')
ax.set_ylabel('Walltime (s)', color="tab:blue")
ax.legend(loc='lower left')

ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
ax2.set_ylabel('Efficiency', color="green")  # we already handled the x-label with ax1
ax2.plot(processes, efficiency_omp, color="green", label='omp')
ax2.plot(processes, efficiency_omp_blas, color='tab:green', label='omp+blas')
for i, proc in enumerate(efficiency_omp[:5]):
    plt.annotate('%0.3f' % proc, xy=(processes[i], efficiency_omp[i]), xytext=(7, 2), textcoords='offset points', color='green')
for i, proc in enumerate(efficiency_omp_blas[:5]):
    plt.annotate('%0.3f' % proc, xy=(processes[i], efficiency_omp_blas[i]), xytext=(7, 2), textcoords='offset points', color='tab:green')
ax2.set_ylim([0, 1.3])
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig('plot_strong_scaling_omp_blas.pdf', format='pdf')
plt.show()