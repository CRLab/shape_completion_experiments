# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import argparse


def plot_stat(stat_file):
    stat = np.loadtxt(stat_file)
    plt.plot(range(len(stat)), stat)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Plot Loss and Error')
    parser.add_argument('--stat_file', default="results/y15_m11_d05_h17_m26/error.txt",
                        help='filepath of stat to plot')

    params = parser.parse_args()

    plot_stat(params.stat_file)


if __name__ == "__main__":
    main()
