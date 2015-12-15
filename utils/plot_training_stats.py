# coding: utf-8

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_stat(stat_files, title, save_file):
    plt.title(title)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, stat_file in enumerate(stat_files):
        color = colors[i]
        print color + ': ' + os.path.split(os.path.split(stat_file)[0])[1]
        stat = np.loadtxt(stat_file)
        plt.plot(range(len(stat)), stat, color)
    plt.savefig(save_file)
    plt.show()
    print ''


def main():

    #results_dir = '/home/avinash/research/shape_completion/train/shape_completion_experiments/experiments/results/y15_m12_d02_h20_m03/'
    results_dir = '/home/avinash/research/shape_completion/train/shape_completion_experiments/experiments/results/y15_m12_d06_h15_m01/'

    jaccard_trained_views = []
    jaccard_holdout_views = []
    jaccard_holdout_models = []

    dir_list = os.listdir(results_dir)
    dir_list.sort()
    for dir_name in dir_list:
        if os.path.isdir(results_dir + dir_name):
            jaccard_trained_views.append(results_dir + dir_name + '/jaccard_err_trained_views.txt')
            jaccard_holdout_views.append(results_dir + dir_name + '/jaccard_err_holdout_views.txt')
            jaccard_holdout_models.append(results_dir + dir_name + '/jaccard_err_holdout_models.txt')

    print ''
    plot_stat(jaccard_trained_views, 'trained_views', results_dir + 'jaccard_err_trained_views.png')
    plot_stat(jaccard_holdout_views, 'holdout_views', results_dir + 'jaccard_err_holdout_views.png')
    plot_stat(jaccard_holdout_models, 'holdout_models', results_dir + 'jaccard_err_holdout_models.png')


if __name__ == "__main__":
    main()
