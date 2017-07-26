import os, itertools
import numpy as np
import matplotlib.pyplot as plt
import utility_functions as util

train_sizes = ["50", "60", "70"]
methods = ["all", "pca", "lda"]
features = util.get_all_features()
classifiers = util.get_classifier_names()

def plot_confusion_matrix(confusion_matrix, genres):
    plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    tick_marks = np.arange(len(genres))
    plt.xticks(tick_marks, genres, rotation=45)
    plt.yticks(tick_marks, genres)
    thresh = confusion_matrix.max() / 2.0
    for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
        plt.text(j, i, confusion_matrix[i, j], horizontalalignment="center", color="white" if confusion_matrix[i, j] > thresh else "black")
    plt.tight_layout()

def plot_all_matrices():
    for feature in features:
        for train_size in train_sizes:
            for classifier in classifiers:
                for method in methods:
                    print feature, train_size, classifier, method
                    base_filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "confusion_matrix")
                    if not os.path.isfile(base_filename + ".png"):
                        plt.figure()
                        plot_confusion_matrix(np.loadtxt(base_filename + ".txt"), util.genres)
                        plt.savefig(base_filename + ".png")
                        plt.close("all")

def main():
    plot_all_matrices()

if __name__ == "__main__":
	main()
