import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 18})

processes = np.array([4, 6, 8, 10, 12, 14, 16])


runtime2 = np.array([
3.056,
2.564,
2.294,
1.939,
1.739,
1.677,
1.620
])

fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(processes, runtime2, 'o-')
ax.set_xscale('log', base=2)
ax.set_yscale('log', base=10)
ax.set_xticks(processes)

ax.set_xticklabels(['4', '6', '8', '10', '12', '14', '16'])
for i, proc in enumerate(runtime2):
    plt.annotate('%0.3f' % proc, xy=(processes[i], runtime2[i]), xytext=(7, 2), textcoords='offset points', color='tab:blue')
ax.set_title('Multi-GPU CG with 70000x70000 Matrix (Strong Scaling)')
ax.set_xlabel('Number of GPUs')
ax.set_ylabel('Walltime (s)', color="tab:blue")
ax.set_xlim((3.9, 18.3))

plt.tight_layout()
plt.savefig('plot_strong_scaling_mpi.pdf', format='pdf')
plt.show()