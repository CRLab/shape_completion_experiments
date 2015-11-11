import argparse
import sys
import numpy as np
import cPickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_pkl_data(data_filepath):
    f = open(data_filepath, 'r')
    return cPickle.load(f)


def gen_plots(x_data, y_actual, y_expected):
    # import IPython
    # IPython.embed()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('x_data')
    Axes3D.scatter(ax, x_data.nonzero()[0], x_data.nonzero()[1], x_data.nonzero()[2])
    bias = .15
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('y_actual')
    Axes3D.scatter(ax, np.round(y_actual + bias).nonzero()[0], np.round(y_actual + bias).nonzero()[1],
                   np.round(y_actual + bias).nonzero()[2])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.title('y_expected')
    Axes3D.scatter(ax, y_expected.nonzero()[0], y_expected.nonzero()[1], y_expected.nonzero()[2])

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file', type=str, help='weight file to visualize')
    args = parser.parse_args(sys.argv[1:])

    x_data, y_actual, y_expected = load_pkl_data(args.data_file)

    gen_plots(x_data, y_actual, y_expected)


if __name__ == "__main__":
    main()
