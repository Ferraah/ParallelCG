import argparse
import numpy as np
import matplotlib.pyplot as plt

def plot_data(file_path):
    # Load data from the text file using numpy
    data = np.loadtxt(file_path, delimiter='\t', skiprows=0)

    # Extract columns
    size = data[:, 0]
    column2 = data[:, 1]
    column3 = data[:, 2]
    column4 = data[:, 3]

    # Plot the data
    plt.plot(size, column2, label='Parallel time')
    plt.plot(size, column3, label='Sequential time')
    plt.plot(size, column4, label='Speedup')

    plt.xlabel('Size of the Problem')
    plt.ylabel('Values')
    plt.title('Line Chart of Data')
    plt.legend()
    plt.savefig('line_chart.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot data from a text file.')
    parser.add_argument('file_path', type=str, help='Path to the data file (tab-separated text)')

    args = parser.parse_args()
    plot_data(args.file_path)