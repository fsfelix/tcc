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

    data_dirs = util.choose_species(num_species)
    data_dirs = util.check_num_files(data_dirs, song_or_call, num_species, num_min)

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

def generate_experiments_parallel(num_species, song_or_call = 'song', scoring = 'f1_weighted'):
    n_global = 4

    data_dirs = util.choose_species(num_species)
    data_dirs = util.check_num_files(data_dirs, song_or_call, num_species, num_min)

#    file_exp = open(util.EXPERIMENTS_DIR + '/parallel_' + 'num_specie_' + str(num_species) + '_num_min_' + str(num_min) + '_scoring_' + scoring + generate_exp_file(), "w+")

    for version in util.VERSIONS_EXPERIMENTS:
        i = 0
        table = create_table(util.FEATURES)
        for feat in util.FEATURES:
            #file_exp.write(feat)
            print('Feature: {} | Version: {}'.format(feat, version))
            #file_exp.write('lol\n')
            #file_exp.flush()

            labels_dict, labels, data = generate_global_features(n_global, feat, data_dirs, song_or_call, util.GLOBAL_FUNCTIONS, version = version)

            clf     = neighbors.KNeighborsClassifier(3, weights = 'uniform')
            scores  = cross_val_score(clf, data, labels, n_jobs = 1, cv = 5, scoring=scoring)
            generate_results_table(table[i], 'kNN', scoring, scores)

            # naïve-bayes
            gnb    = GaussianNB()
            scores = cross_val_score(gnb, data, labels, n_jobs = 1, cv = 5, scoring=scoring)
            generate_results_table(table[i], 'GaussianNB', scoring, scores)

            # SVM
            #clf = svm.SVC(kernel = 'rbf', C = 1)
            #clf = svm.SVC(kernel = 'poly', C = 1)

            clf = svm.SVC(kernel = 'linear', max_iter = 100000000 ,C = 1, decision_function_shape='ovr')
            #file_exp.write(str(clf) + '\n')
            #file_exp.flush()
            scores = cross_val_score(clf, data, labels, n_jobs = 1, cv = 5, scoring=scoring)
            generate_results_table(table[i], 'SVM', scoring, scores)
            i += 1

        # lock.acquire()
        #file_exp.write("Número de espécies: {}\n".format(num_species))
        #file_exp.write("Diretórios: \n")
        print("Número de espécies: {} Número Mínimo: {} Scoring: {}".format(num_species, num_min, scoring))
        for dir in data_dirs:
            print(dir)
            #file_exp.write("{} \n".format(dir))
        #file_exp.write("\n")
        #file_exp.write("type of score: {}\n".format(scoring))
        #file_exp.write('Type of recording: ' + str(version) + '\n')
        print_table(table)
        #write_table(table, file_exp)
        #file_exp.flush()

    #file_exp.close()

def generate_exp_file():
    return 'experiment_' + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')

def init(l):
    global lock
    lock = l

def experiments_parallel(num_exp, num_cores):
    l = Lock()

    if num_cores == -2:
        num_cores = cpu_count()

    pool = Pool(initializer = init, initargs = (l,), processes = num_cores)

    args_p = [3, 5, 8, 12, 20] * num_exp
    # args_p = [3] * num_exp

    pool.map(generate_experiments_parallel, args_p)
    pool.close()
    pool.join()

def main():
    global num_min
    # num_exp = int(input("número de experimentos: "))
    # num_min = int(input("número minimo de arquivos por especie: "))
    # num_cores = int(input("número de cores (-1 sem paralelismo, -2 número máximo possível): "))

    num_exp = 5
    num_min = 30
    num_cores = -2

    if num_cores != -1:
        experiments_parallel(num_exp, num_cores)

    else:
        #num_species = [3, 5, 8, 12, 20]
        num_species = [3]
        file_exp = open(util.EXPERIMENTS_DIR + '/' + generate_exp_file(), "w+")
        for num in num_species:
            file_exp.write('Número de espécies: {}\n'.format(num))
            for i in range(num_exp):
                print("Número espécie: {} | Exp: {}/{}".format(num, i + 1, num_exp))
                generate_experiments(num, file_exp, song_or_call = 'song')
        file_exp.close()

if __name__ == '__main__':
    main()
