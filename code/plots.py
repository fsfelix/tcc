import os 
import librosa
import librosa.display
import importlib
import matplotlib
import numpy as np
import string as strp
import pr_util as util
import matplotlib.pyplot as plt
import generate_global_features_2 as ggf
import extract_syllable_duration_4 as esd

from matplotlib.backends.backend_pdf import PdfPages

#from extract_syllable_duration_4 import autodetec, dilate_both_and

importlib.reload(util)
importlib.reload(ggf)
importlib.reload(esd)

def matrix_from_table(table_dir):
    file_table = open(table_dir)
    matrix = []
    all_matrix = []
    for l in file_table.readlines():
        if l.count('filtered') == 1:
            all_matrix.append(matrix)
            matrix = []
        else:
            l = l.replace(' ', '')
            ls = l.split('|')
            if (ls[0] != '' and ls[0] != 'mfcc' and ls[0] != 'syllable_dur'):
                feat = ls[0]
                knn = float(ls[1].split('(')[0])
                nb  = float(ls[2].split('(')[0])
                svm = float(ls[3].split('(')[0])
                matrix.append([knn, nb, svm])
    all_matrix = np.array(all_matrix)
    return all_matrix

def color_plot_data(data_all, title_global):
    plt.figure(figsize=(10, 10))
    plt.suptitle(title_global, fontsize = 20, y = 1.05)
    n_lin = 2
    n_col = 2
    titles = ['sem filtragem', 'filtro 1', 'filtro 2', 'filtro3']
    for i in range(n_lin):
        for j in range(n_col):
            n_fig = i * n_col + (j + 1)
            data = data_all[n_fig -1 ]
            ax = plt.subplot(n_lin, n_col, n_fig)
            heatmap = ax.pcolor(data, cmap=plt.cm.Reds)

            ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            ax.set_xticklabels(util.CLASSIFIERS, minor=False)
            ax.set_yticklabels(util.FEATURES_PLOT, minor=False)
            plt.colorbar(mappable=heatmap, ax=ax)
            plt.title("{}\n\n".format(titles[n_fig - 1]))

    plt.tight_layout()

def line_plot_data(data_all, n_lin, n_col, title_global, titles, xlabel, ylabel, legend, xticks):
    fig = plt.figure(figsize=(10*n_lin, 5*n_col))
    fig.suptitle(title_global, fontsize = 30, y = 1.05)
    for i in range(n_lin):
        for j in range(n_col):
            n_fig = i * n_col + (j + 1)
            data = data_all[n_fig -1 ]
            ax = plt.subplot(n_lin, n_col, n_fig)
            ax.set_ylim((0.0, 0.8))
            ax.plot(data)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(legend)
            plt.xticks(np.arange(0,len(xticks),1),xticks)
            plt.title("{}".format(titles[n_fig - 1]))
    fig.tight_layout()

def transpose_matrices(all_data):
    all_data_t = []
    for d in all_data:
        all_data_t.append(d.T)
    all_data_t = np.array(all_data_t)
    return all_data_t
