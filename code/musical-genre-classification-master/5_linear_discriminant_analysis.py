import os
import numpy as np
import utility_functions as util
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def lda(feature, split, classifier, X_train, X_test, y_train, y_test):
	max_hit_rate, max_n_components = util.check_all_n_components(classifier[0],
		X_train, X_test, y_train, y_test, LinearDiscriminantAnalysis, X_train, y_train)
	directory = os.path.join(util.RESULTS_DIR, feature, split.split("/")[0], classifier[1], "lda")

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
			util.prepare_splits_and_call_function(feature, classifier, lda)

			print feature + "_beat", classifier[1]
			util.prepare_splits_and_call_function(feature + "_beat", classifier, lda)

if __name__ == "__main__":
	main()
