import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def parabula(x, a):
    return a * x**2#+ b * x + c

def plot_data(file_path, title):

    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 13.5})
    plt.yscale('log')
    plt.xscale('log')
    # Load data from the text file using numpy
    data = np.loadtxt(file_path, delimiter='\t', skiprows=0)

    # Extract columns
    size = data[:, 0]
    column2 = data[:, 1]
    column3 = data[:, 2]
    column4 = data[:, 3]

    # x-data for fit
    size_fit = np.linspace(size[0], size[-1], 1000)
    # find fit params
    popt2, pcov2 = curve_fit(parabula, size, column2)
    popt3, pcov3 = curve_fit(parabula, size, column3)
    # plot fit func
    plt.plot(size_fit, parabula(size_fit, *popt2), linewidth=0.9, color='tab:green', label=rf'Fit func parallel: $a \cdot n^2$, $a = ${np.format_float_scientific(popt2[0], precision=3, exp_digits=1)}')
    plt.plot(size_fit, parabula(size_fit, *popt3), linewidth=0.9, color='tab:red', label=rf'Fit func sequential: $a \cdot n^2$, $a = ${np.format_float_scientific(popt3[0], precision=3, exp_digits=1)}')

    # Plot the data
    plt.plot(size, column2, label='Parallel time', linewidth=0.9, color='green', marker='o')
    plt.plot(size, column3, label='Sequential time', linewidth=0.9, color='red', marker='o')
    # plt.plot(size, column4, label='Speedup')

    # Annotate key points with values
    for i, txt in enumerate(column2):
        plt.annotate(f'{txt:.2f}', (size[i], column2[i]), textcoords="offset points", xytext=(20,-15), ha='center')

    for i, txt in enumerate(column3):
        plt.annotate(f'{txt:.2f}', (size[i], column3[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.xlabel('Matrix size')
    plt.ylabel('Walltime (s)')
    plt.title(title)
    plt.legend()
    plt.savefig(file_path+'V1'+'.png')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the data file (tab-separated text)')
    parser.add_argument('title', type=str, help='Plot title')

    args = parser.parse_args()
    plot_data(args.file_path, args.title)
