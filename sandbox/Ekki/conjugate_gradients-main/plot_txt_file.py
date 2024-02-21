# Script for plotting the benchmarks
#
# Load python: module load Python
# Set up env: python -m pip install numpy matplotlib
# Execute as: python3 plot_txt_file.py file_name.txt plot_output.pdf
#

import numpy as np
import matplotlib.pyplot as plt
import sys

file = sys.argv[1] # This script takes two additional arguments as input
out = sys.argv[2]

# Read out file content
f = open(file, 'r')
lines = f.readlines()[1:]
f.close()

array = []
for line in lines:
    line = line.strip('\n').split('\t')
    for number in line:
        array.append(float(number))

array = np.array(array)
matrix_sizes = array[0::2].astype(int)
times = array[1::2].astype(float)

# plot
# plt.rcParams['text.usetex'] = True
plt.plot(matrix_sizes, times, marker="+")
plt.title("Benchmark")
plt.xlabel("Matrix size")
plt.ylabel("time [s]")
plt.semilogy()
plt.grid()
plt.savefig(out, format='pdf')