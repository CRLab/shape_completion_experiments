# coding: utf-8

import matplotlib.pyplot as plt
import os
import numpy as np


def plot_stat(stat_files, title, save_file, filter_data=False):
    plt.title(title, fontsize=22)
    plt.xlabel('Number of Training Batches (Thousands)', fontsize=18)
    plt.ylabel('Jaccared Similarity', fontsize=18)
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i, stat_file in enumerate(stat_files):
        color = colors[i]
        print color + ': ' + os.path.split(os.path.split(stat_file)[0])[1]
        label = reformat_string(os.path.split(os.path.split(stat_file)[0])[1])
        stat = np.loadtxt(stat_file)
        if filter_data:
            filter_len = 30
            stat = moving_average(stat, filter_len)
        x = np.array(range(len(stat)))/100.0
        plt.plot(x, stat, color, label=label, linewidth=2)
    plt.ylim((0,1))
    plt.legend(loc=4, fontsize=18, labelspacing=0.2)
    plt.savefig(save_file)
    plt.show()
    print ''


def moving_average(data, filter_len):
    result = np.zeros(len(data))
    for i in range(0, len(data), 1):
        start = max(i - filter_len, 0)
        valid = data[start:i+1]
        valid = valid[~np.isnan(valid)]
        result[i] = np.sum(valid) / len(valid)
    return result


def reformat_string(string):
    string = string.replace('_', ' ')
    string = string.replace('ycb', 'ycb,')
    string = string.replace('all', '19')
    string = string.replace('no', '0')
    return string.upper()


def main():

    #results_dir = '/home/avinash/research/shape_completion/train/shape_completion_experiments/experiments/results/y15_m12_d02_h20_m03_ycb_and_shrec/'
    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m01_d18_h17_m58/'

    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m01_d23_h10_m02/'

    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m01_d22_h11_m57/'

    #RPY YCB FIRST RUN
    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m01_d26_h20_m49/'

    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m01_d28_h22_m48/'
    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m02_d01_h14_m16/'
    #results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m02_d03_h17_m02/'
    results_dir = '/home/jvarley/shape_completion/train/shape_completion_experiments/experiments/results/y16_m07_d26_h18_m46/'

    COMPARING_RUNS = False
    filter_data = True

    jaccard_trained_views = []
    jaccard_holdout_views = []
    jaccard_holdout_models = []

    if COMPARING_RUNS:
        dir_list = os.listdir(results_dir)
        dir_list.sort()
        for dir_name in dir_list:
            if os.path.isdir(results_dir + dir_name):
                jaccard_trained_views.append(results_dir + dir_name + '/jaccard_err_trained_views.txt')
                jaccard_holdout_views.append(results_dir + dir_name + '/jaccard_err_holdout_views.txt')
                jaccard_holdout_models.append(results_dir + dir_name + '/jaccard_err_holdout_models.txt')
    else:
        jaccard_trained_views.append(results_dir + 'jaccard_err_trained_views.txt')
        jaccard_holdout_views.append(results_dir + 'jaccard_err_holdout_views.txt')
        jaccard_holdout_models.append(results_dir + 'jaccard_err_holdout_models.txt')

    print ''
    if filter_data:
        plot_stat(jaccard_trained_views, 'Trained Views',
                  results_dir + 'jaccard_err_trained_views_filtered.pdf', filter_data)
        plot_stat(jaccard_holdout_views, 'Holdout Views',
                  results_dir + 'jaccard_err_holdout_views_filtered.pdf', filter_data)
        plot_stat(jaccard_holdout_models, 'Holdout Models',
                  results_dir + 'jaccard_err_holdout_models_filtered.pdf', filter_data)
    else:
        plot_stat(jaccard_trained_views, 'Trained Views', results_dir + 'jaccard_err_trained_views.pdf')
        plot_stat(jaccard_holdout_views, 'Holdout Views', results_dir + 'jaccard_err_holdout_views.pdf')
        plot_stat(jaccard_holdout_models, 'Holdout Models', results_dir + 'jaccard_err_holdout_models.pdf')



if __name__ == "__main__":
    main()
