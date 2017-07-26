import os
import numpy as np
import utility_functions as util

def all(feature, split, classifier, X_train, X_test, y_train, y_test):
	y_pred = util.classify(classifier[0], X_train, y_train, X_test)
	hit_rate = 100.0 * (y_test == y_pred).sum() / y_test.shape[0]
	directory = os.path.join(util.RESULTS_DIR, feature, split.split("/")[0], classifier[1], "all")

	accuracy = os.path.join(directory, "accuracy.txt")
	if not os.path.isfile(accuracy):
		np.savetxt(accuracy, [hit_rate])

	n_components = os.path.join(directory, "n_components.txt")
	if not os.path.isfile(n_components):
		np.savetxt(n_components, [X_train.shape[1]])

def main():
	for feature in util.features:
		for classifier in util.classifiers:
			print feature, classifier[1]
			util.prepare_splits_and_call_function(feature, classifier, all)

			print feature + "_beat", classifier[1]
			util.prepare_splits_and_call_function(feature + "_beat", classifier, all)

if __name__ == "__main__":
	main()
