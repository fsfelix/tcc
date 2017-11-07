import pr_util as util
import numpy as np
import datetime
import time

from sklearn import svm, neighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from generate_global_features_2 import generate_global_features

from multiprocessing import Pool, Lock, cpu_count, RLock

def create_table(features):
    table = []
    for feat in features:
        table.append([feat])
    return table

def print_table(table):
    print(' | '.rjust(23), end = '')
    for classifier in util.CLASSIFIERS:
        print('{} | '.format(classifier.rjust(20)), end = '')
    print()
    for line in table:
        for element in line:
            print('{} | '.format(str(element).rjust(20)), end = '')
        print()

def write_table(table, file_exp):
    file_exp.write(' | '.rjust(23))
    for classifier in util.CLASSIFIERS:
        file_exp.write('{} | '.format(classifier.rjust(20)))
    file_exp.write('\n')
    for line in table:
        for element in line:
            file_exp.write('{} | '.format(str(element).rjust(20)))
        file_exp.write('\n')
    file_exp.write('\n')

def generate_results_table(table, clf, scoring, scores):
    result = '{0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2)
    table.append(result)
    print('{0} - {1}: {2:.2f} (+/- {3:.2f})'.format(clf, scoring, scores.mean(), scores.std() * 2))
    print(scores)

def generate_experiments(num_species, file_exp, song_or_call = 'song', scoring = 'f1_weighted'):

    n_global = 4

    data_dirs = util.choose_species(num_species, num_min, song_or_call)
    #data_dirs = util.check_num_files(data_dirs, song_or_call, num_species, num_min)

    print("Diretórios: ")
    for dir in data_dirs:
        print(dir)
    print()

    file_exp.write("Diretórios: \n")
    for dir in data_dirs:
        file_exp.write("{} \n".format(dir))
    print()

    file_exp.write("type of score: {}\n".format(scoring))
    table = create_table(util.FEATURES)

    for version in util.VERSIONS_EXPERIMENTS:
        i = 0
        table = create_table(util.FEATURES)
        for feat in util.FEATURES:
            print('Feature: {} | Version: {}'.format(feat, version))
            labels_dict, labels, data = generate_global_features(n_global, feat, data_dirs, song_or_call, util.GLOBAL_FUNCTIONS, version = version)

            clf     = neighbors.KNeighborsClassifier(3, weights = 'uniform')
            scores  = cross_val_score(clf, data, labels, n_jobs = -1, cv = 5, scoring=scoring)
            generate_results_table(table[i], 'kNN', scoring, scores)

            # naïve-bayes
            gnb    = GaussianNB()
            scores = cross_val_score(gnb, data, labels, n_jobs = -1, cv = 5, scoring=scoring)
            generate_results_table(table[i], 'GaussianNB', scoring, scores)

            # SVM
            #clf = svm.SVC(kernel = 'rbf', C = 1)
            #clf = svm.SVC(kernel = 'poly', C = 1)

            clf = svm.SVC(kernel = 'linear', C = 1, decision_function_shape='ovr')
            file_exp.write(str(clf) + '\n')
            scores = cross_val_score(clf, data, labels, n_jobs = -1, cv = 5, scoring=scoring)
            generate_results_table(table[i], 'SVM', scoring, scores)

            print()
            i += 1

        file_exp.write('Type of recording: ' + str(version) + '\n')
        print_table(table)
        write_table(table, file_exp)

def generate_exp_file(num_exp, num_min, song_or_call, spc):
    spcs = ''
    for s in spc:
        spcs += '_' + str(s)

    spcs += '_'

    return 'experiment-' + 'spcs' + spcs + '-' + 'numexps-' + str(num_exp) + '-num_min-' + str(num_min) + '-' + str(song_or_call) + '-' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

# info[0] = feature
# info[1] = version
# info[2] = data_dirs
# info[3] = song_or_call
# info[4] = n_species
# info[5] = scoring

def generate_experiment(info):
    n_function_global =  4

    feat         = info[0]
    version      = info[1]
    data_dirs    = info[2]
    song_or_call = info[3]
    n_species    = info[4]
    scoring      = info[5]
    num_min      = info[6]
    num_exp      = info[7]
    num_max      = info[8]
    kernel       = 'linear'
    k            = 3
    cv           = 5

    print("{} {} {} {} {}".format(feat, version, data_dirs, song_or_call, n_species))
    print(data_dirs)
    labels_dict, labels, data = generate_global_features(n_function_global, feat, data_dirs, song_or_call, util.GLOBAL_FUNCTIONS, version = version)

    len_data = len(data)
    len_labels = len(labels)
    #print("len data: {}".format(len_data))
    #print("len labels: {}".format(len_labels))

    if len_data != len_labels or data == [] or len_data == 0 or len_labels == 0:
        print("ACHAMOS UMA INCOSISTENCIA")
        print(data_dirs)
        resp = dict(n_species = n_species,  feat = feat, version = version, dirs = data_dirs, song_or_call = song_or_call, scoring = scoring, knn = '-1', gnb = '-1', svm = '-1', num_min = num_min, num_max = num_max, num_exp = num_exp, kernel = kernel, kNN = k, cv = cv)

    else:
        #print("kNN Starting ->")
        clf         = neighbors.KNeighborsClassifier(k, weights = 'uniform')
        scores      = cross_val_score(clf, data, labels, n_jobs = 1, cv = cv, scoring=scoring)
        result_knn  = '{0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2)
        #print("[DONE] kNN Done <-")

        #print("GNB Starting ->")
        gnb        = GaussianNB()
        scores     = cross_val_score(gnb, data, labels, n_jobs = 1, cv = cv, scoring=scoring)
        result_gnb = '{0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2)
        #print("[DONE] GNB Done <-")

        #print("SVM Starting ->")
        clf        = svm.SVC(kernel = kernel, C = 1, decision_function_shape='ovr')
        scores     = cross_val_score(clf, data, labels, n_jobs = 1, cv = cv, scoring=scoring)
        result_svm = '{0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2)
        #print("[DONE] SVM Done <-")

        resp = dict(n_species = n_species,  feat = feat, version = version, dirs = data_dirs, song_or_call = song_or_call, scoring = scoring, knn = result_knn, gnb = result_gnb, svm = result_svm, num_min = num_min, num_max = num_max, num_exp = num_exp, kernel = kernel, kNN = k, cv = cv)

    return resp


def generate_info(num_species, num_exp, num_min, num_max, song_or_call):
    print("Generating infos for parallel...")
    infos = []
    for n in range(num_exp):
        for spc in num_species:
            if num_min == -1:
                DIRS = util.return_n_most_frequent_species(spc, song_or_call)
            else:
                DIRS = util.choose_species(spc, num_min, num_max, song_or_call)
            for feat in util.FEATURES:
                for version in util.VERSIONS_EXPERIMENTS:
                    print((feat, version, DIRS, song_or_call, spc, 'f1_weighted'))
                    infos.append((feat, version, DIRS, song_or_call, spc, 'f1_weighted', num_min, n + 1, num_max))
    print("[DONE] Info generated.")
    print("Info lenght: {}".format(len(infos)))
    return infos

def write_info(d, file_exp):
    file_exp.write('\n--------------------------------------\n')
    file_exp.write('experimento número: {} \n'.format(d['num_exp']))
    file_exp.write('numero de especies: {}\n'.format(d['n_species']))
    file_exp.write('numero minimo de arquivos por especie (-1 most frequent): {}\n'.format(d['num_min']))
    file_exp.write('numero máximo de arquivos por especie: {}\n'.format(d['num_max']))
    file_exp.write('diretórios: {}\n'.format(d['dirs']))
    file_exp.write('scoring: {}\n'.format(d['scoring']))
    file_exp.write('song_or_call: {}\n'.format(d['song_or_call']))
    file_exp.write('versão: {}\n'.format(d['version']))
    file_exp.write('kernel SVM: {}\n'.format(d['kernel']))
    file_exp.write('k do kNN: {}\n'.format(d['kNN']))
    file_exp.write('numero cross-validation: {}'.format(d['cv']))
    file_exp.write('\n--------------------------------------\n')

def tables_from_dicts(dicts, file_exp):

    while len(dicts) > 0:
        lines = []
        ds = []
        current = dicts[0]

        for d in dicts:
            if d['dirs'] == current['dirs'] and d['version'] == current['version'] and d['num_exp'] == current['num_exp']:
                ds.append(d)

        for d in ds:
            dicts.remove(d)
            lines.append([d['feat'], d['knn'], d['gnb'], d['svm']])

        print_table(lines)
        write_info(ds[0], file_exp)
        write_table(lines, file_exp)

def experiments_parallel(num_exp, num_cores, num_min, num_max, song_or_call, spc):

    if num_cores == -2:
        num_cores = cpu_count()

    pool = Pool(num_cores)

    DIR = util.EXPERIMENTS_DIR + '/' + generate_exp_file(num_exp, num_min, song_or_call, spc)

    song_or_call = 'song'

    infos = generate_info(spc, num_exp, num_min, num_max, song_or_call)

    dicts = []

    with open(DIR, 'w') as f:
        for result in pool.imap(generate_experiment, infos):
            f.write(str(result))
            f.write('\n')
            f.write('\n')
            f.write('\n')
            dicts.append(result)
        pool.close()
        pool.join()
        tables_from_dicts(dicts, f)


def main():

    num_exp      = int(input("número de experimentos: "))
    spc          = str(input("lista com número de espécies separado por espaços (ex: 3 5 8): "))
    num_min      = int(input("número minimo de arquivos por especie (-1 para as n especies + freq): "))
    num_max      = int(input("número maximo de arquivos por especie: "))
    num_cores    = int(input("número de cores (-1 sem paralelismo, -2 número máximo possível): "))
    song_or_call = str(input("song or call: "))

    spc = [int(n) for n in spc.split(' ')]

    if num_cores != -1:
        experiments_parallel(num_exp, num_cores, num_min, num_max, song_or_call, spc)

    # else:
    #     #num_species = [3, 5, 8, 12, 20]
    #     num_species = [3]
    #     file_exp = open(util.EXPERIMENTS_DIR + '/' + generate_exp_file(), "w+")
    #     for num in num_species:
    #         file_exp.write('Número de espécies: {}\n'.format(num))
    #         for i in range(num_exp):
    #             print("Número espécie: {} | Exp: {}/{}".format(num, i + 1, num_exp))
    #             generate_experiments(num, file_exp, song_or_call = 'song')
    #     file_exp.close()

if __name__ == '__main__':
    main()
