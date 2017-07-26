import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import utility_functions as util

def local_func(feature, split, classifier, X_train, X_test, y_train, y_test, data_type):
	if classifier[2] is None:
		y_pred = util.classify(classifier[0], X_train, y_train, X_test)
	else:
		y_pred = util.grid_search(classifier, X_train, y_train, X_test)

	directory = os.path.join(util.RESULTS_DIR, feature, split.split("/")[0], classifier[1], data_type)
	util.save_data_to_txt(os.path.join(directory, "f1_macro.txt"), [f1_score(y_test, y_pred, average='macro')])
	util.save_data_to_txt(os.path.join(directory, "confusion_matrix.txt"), confusion_matrix(y_test, y_pred))

def global_func(feature, split, classifier, X_train, X_test, y_train, y_test):
	directory = os.path.join(util.RESULTS_DIR, feature, split.split("/")[0], classifier[1])

	all_f1_macro = os.path.join(directory, "all", "f1_macro.txt")
	if not os.path.isfile(all_f1_macro):
		local_func(feature, split, classifier, X_train, X_test, y_train, y_test, "all")

	pca_f1_macro = os.path.join(directory, "pca", "f1_macro.txt")
	if not os.path.isfile(pca_f1_macro):
		pca_n_components = int(np.loadtxt(os.path.join(directory, "pca", "n_components.txt")))
		X_train_pca, X_test_pca = util.transform_data(X_train, X_test, pca_n_components, PCA, X_train)
		local_func(feature, split, classifier, X_train_pca, X_test_pca, y_train, y_test, "pca")

	lda_f1_macro = os.path.join(directory, "lda", "f1_macro.txt")
	if not os.path.isfile(lda_f1_macro):
		lda_n_components = int(np.loadtxt(os.path.join(directory, "lda", "n_components.txt")))
		X_train_lda, X_test_lda = util.transform_data(X_train, X_test, lda_n_components, LinearDiscriminantAnalysis, X_train, y_train)
		local_func(feature, split, classifier, X_train_lda, X_test_lda, y_train, y_test, "lda")

def main():
	for feature in util.features:
		print feature
		for classifier in util.classifiers:
			print classifier[1]
			util.prepare_splits_and_call_function(feature, classifier, global_func)
			util.prepare_splits_and_call_function(feature + "_beat", classifier, global_func)

if __name__ == "__main__":
	main()
