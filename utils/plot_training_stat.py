# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_stat(stat_files):

    for i, stat_file in enumerate(stat_files.split(',')):
        stat = np.loadtxt(stat_file)
        plt.plot(range(len(stat)), stat, label=str(i))

    plt.legend(loc=4)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Loss and Error')
    parser.add_argument('--stat_files',
                        default="results/y15_m11_d05_h17_m26/error.txt",
                        help='filepath of stat to plot')

    params = parser.parse_args()

    plot_stat(params.stat_files)


if __name__ == "__main__":
    main()
