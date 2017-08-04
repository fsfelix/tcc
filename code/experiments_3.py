import pr_util as util
import numpy as np

from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from generate_global_features_2 import generate_global_features

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

def check_num_files(data_dirs, song_or_call, num_species, n_min):
    # Check if all dirs have at least n_min files

    for i in range(num_species):
        if util.num_files([data_dirs[i]], song_or_call) < n_min:
            print("loop infinito?")
            data_dirs = util.choose_species(num_species)
            i = 0

    return data_dirs

def main():
    num_species  = int(input('Número de espécies: '))
    song_or_call = 'song'

    n_min    = 5
    n_global = 4

    data_dirs = util.choose_species(num_species)
    check_num_files(data_dirs, song_or_call, num_species, n_min)

    print("Diretórios: ")
    for dir in data_dirs:
        print(dir)
    print()


    table = create_table(util.FEATURES)

    i = 0
    for feat in util.FEATURES:
        print('Feature: {}'.format(feat))
        labels_dict, labels, data = generate_global_features(n_global, feat, data_dirs, song_or_call, util.GLOBAL_FUNCTIONS)

        # kNN
        res, max_k = util.kNN(data, labels, range(3, 4), 5)
        table[i].append(res)
        print('kNN: Accuracy: {} | k: {}'.format(res, max_k))

        # naïve-bayes
        gnb = GaussianNB()
        scores = cross_val_score(gnb, data, labels, cv = 5)
        acc = '{0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2)
        table[i].append(acc)
        print('GaussianNB: Accuracy: {0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2))
        print(scores)

        # SVM
        clf = svm.SVC(kernel = 'linear', C = 1)
        scores = cross_val_score(clf, data, labels, cv = 5)
        acc = '{0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2)
        table[i].append(acc)
        print('SVM: Accuracy: {0:.2f} (+/- {1:.2f})'.format(scores.mean(), scores.std() * 2))
        print(scores)

        i += 1

    print_table(table)
if __name__ == '__main__':
    main()
