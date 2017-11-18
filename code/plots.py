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

from collections import OrderedDict
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

def color_plot_data_f_c(data_all, title_global):
    plt.figure(figsize=(10, 10))
    plt.suptitle(title_global, fontsize = 20)
    n_lin = 2
    n_col = 2
    titles = ['sem filtragem', 'filtro 1', 'filtro 2', 'filtro3']
    for i in range(n_lin):
        for j in range(n_col):
            n_fig = i * n_col + (j + 1)
            data = data_all[n_fig -1 ]
            ax = plt.subplot(n_lin, n_col, n_fig)
            heatmap = ax.pcolor(data, cmap=plt.cm.Reds, vmin=0, vmax=0.8)

            ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            ax.set_xticklabels(util.CLASSIFIERS, minor=False)
            ax.set_yticklabels(util.FEATURES_PLOT, minor=False)
            plt.colorbar(mappable=heatmap, ax=ax)
            plt.title("{}\n\n".format(titles[n_fig - 1]))

    plot = plt.tight_layout()
    plt.subplots_adjust(top=0.88)

    return plot

def color_plot_data_f_v(data_all, title_global):
    data_all = data_all.T
    plt.figure(figsize=(15, 5))
    plt.suptitle(title_global, fontsize = 20)
    n_lin = 1
    n_col = 3
    titles = ['sem filtragem', 'filtro 1', 'filtro 2', 'filtro3']
    for i in range(n_lin):
        for j in range(n_col):
            n_fig = i * n_col + (j + 1)
            data = data_all[n_fig -1 ]
            ax = plt.subplot(n_lin, n_col, n_fig)
            heatmap = ax.pcolor(data, cmap=plt.cm.Blues, vmin=0, vmax=0.8)

            ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
            ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)

            # want a more natural, table-like display
            ax.invert_yaxis()
            ax.xaxis.tick_top()

            ax.set_xticklabels(titles, minor=False)
            ax.set_yticklabels(util.FEATURES_PLOT, minor=False)
            plt.colorbar(mappable=heatmap, ax=ax)
            plt.title("{}\n\n".format(util.CLASSIFIERS[n_fig - 1]))
    plot = plt.tight_layout()
    plt.subplots_adjust(top=0.78)

    return plot

def line_plot_data(data_all, n_lin, n_col, title_global, titles, xlabel, ylabel, legend, xticks):
    fig = plt.figure(figsize=(10*n_lin, 5*n_col))
    fig.suptitle(title_global, fontsize = 20)
    for i in range(n_lin):
        for j in range(n_col):
            n_fig = i * n_col + (j + 1)
            data = data_all[n_fig -1 ]
            ax = plt.subplot(n_lin, n_col, n_fig)
            ax.set_ylim((0.0, 0.8))
            #print(data)
            ax.plot(data)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.legend(legend)
            plt.xticks(np.arange(0,len(xticks),1),xticks)
            plt.title("{}".format(titles[n_fig - 1]))
    plot = plt.tight_layout()

    return plot



def line_plot_data_max(data_all, n_lin, n_col, title_global, titles, xlabel, ylabel, legend, xticks, dict_markers, axis):
    fig = plt.figure(figsize=(15*n_lin, 7.5*n_col))
    my_suptitle = fig.suptitle(title_global + '           ', fontsize = 20)
    markers=['o','X','D','8','p','*','+','8','+','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
    colors = ['r', 'g', 'b', 'c', 'm', 'y','k']
    m_i = 0
    f_i = 0
    c_i = 0
    classifier = np.argmax(data_all[0], axis=axis).flatten('F')
    data_all[0] = np.max(data_all[0], axis=axis)
    #classifier = [0,1,2]*100
    cl = dict_markers
    for i in range(n_lin):
        for j in range(n_col):
            n_fig = i * n_col + (j + 1)
            data = data_all[n_fig -1 ]
            ax = plt.subplot(n_lin, n_col, n_fig)
            ax.set_ylim((np.min(data)-0.01, np.max(data)+0.01))
            for c in cl:
                p2, = ax.plot(0, 0, marker=markers[c], markersize=12, color='k', alpha=0.4)
                p2.set_label(cl[c])
                h, l = ax.get_legend_handles_labels()
                by_label = OrderedDict(zip(l, h))
                ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)

            #print("data \n {}".format(data))
            #print("data.T \n {}".format(data.T))
            for d in data.T:
                #print("d \n {}".format(d))
                N = len(d)
                for i in range(N):
                    #print("c_i: {} feat: {} valor: {} marker: {} color: {}".format(c_i, util.FEATURES_PLOT[f_i], d[i], classifier[c_i], colors[m_i]))
                    p1, = ax.plot(i, d[i], marker=markers[classifier[c_i]], markersize=15,color=colors[m_i], alpha=.3)
                    if i != N-1:
                        line, = ax.plot([i, i+1], [d[i], d[i+1]], color=colors[m_i])
                        line.set_label(legend[f_i])
                    h, l = ax.get_legend_handles_labels()
                    by_label = OrderedDict(zip(l, h))
                    ax.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1, 0.5), numpoints=1)
                    c_i += 1
                f_i += 1
                m_i += 1
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xticks(np.arange(0,len(xticks),1),xticks)
            plt.title("{}".format(titles[n_fig - 1]))
    plot = plt.tight_layout()
    plt.subplots_adjust(top=0.88, right=0.90)

    #fig.savefig('teste.pdf', dpi=300, bbox_inches='tight',bbox_extra_artists=[my_suptitle])

    return plot

def max_f_per_version(all_data):
    data_f = []
    for data in all_data:
        data_f.append(np.max(data,axis=1))
    data_f = np.array(data_f)
    return data_f

def line_plot_max(all_data, title):
    return line_plot_data_max([all_data], 1, 1, title,['max(f-measure) por versão/feat'], 'Versão filtrada', 'F-measure', util.FEATURES_PLOT, ['no filter', 'filter1', 'filter2', 'filter3'],{0:'kNN', 1:'NB', 2:'SVM'}, 2)

def line_plot_max_v_c(all_data, title):
    return line_plot_data_max([all_data], 1, 1, title,['max(f-measure) por versão/classificador'], 'Versão filtrada', 'F-measure', util.CLASSIFIERS, ['no filter', 'filter1', 'filter2', 'filter3'],{0:'rmse', 1:'mfcc', 2:'spec_band', 3:'spec_cent', 4:'spec_roll', 5:'syllable_dur', 6:'zcr'}, 1)

def transpose_matrices(all_data):
    all_data_t = []
    for d in all_data:
        all_data_t.append(d.T)
    all_data_t = np.array(all_data_t)
    return all_data_t

def plot_all(all_data, title):
    color_plot_data_f_c(all_data,title)
    color_plot_data_f_v(all_data,title)
    line_plot_data(all_data, 2, 2, title,['no filter', 'filter1', 'filter2', 'filter3'], 'Feature', 'F-measure', util.CLASSIFIERS, util.FEATURES_PLOT )
    line_plot_max(all_data, title)
    line_plot_max_v_c(all_data, title)


