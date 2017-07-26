import os
import numpy as np
import utility_functions as util
from sklearn.decomposition import PCA

def pca(feature, split, classifier, X_train, X_test, y_train, y_test):
	max_hit_rate, max_n_components = util.check_all_n_components(classifier[0],
		X_train, X_test, y_train, y_test, PCA, X_train)
	directory = os.path.join(util.RESULTS_DIR, feature, split.split("/")[0], classifier[1], "pca")

	accuracy = os.path.join(directory, "accuracy.txt")
	if not os.path.isfile(accuracy):
		np.savetxt(accuracy, [max_hit_rate])

	n_components = os.path.join(directory, "n_components.txt")
	if not os.path.isfile(n_components):
		np.savetxt(n_components, [max_n_components])

def main():
	for feature in util.features:
		for classifier in util.classifiers:
			print feature, classifier[1]
			util.prepare_splits_and_call_function(feature, classifier, pca)

			print feature + "_beat", classifier[1]
			util.prepare_splits_and_call_function(feature + "_beat", classifier, pca)

if __name__ == "__main__":
	main()
