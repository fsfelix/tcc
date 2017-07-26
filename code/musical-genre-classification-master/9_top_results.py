import os, operator
import numpy as np
import utility_functions as util

train_sizes = ["50", "60", "70"]
methods = ["all", "pca", "lda"]
features = util.get_all_features()
classifiers = util.get_classifier_names()

def top_results_overall():
	print "top results overall:"
	results = []
	for feature in features:
		for classifier in classifiers:
			for train_size in train_sizes:
				for method in methods:
					filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "f1_macro.txt")
					results.append([filename, np.loadtxt(filename)])

	results = sorted(results, key=operator.itemgetter(1))

	for i in range(len(results) - 10, len(results)):
		info = results[i][0].split("/")
		print info[1], info[2], info[3], info[4], results[i][1]

def top_results_by_feature():
	print "top results by feature:"
	for feature in features:
		results = []
		for classifier in classifiers:
			for train_size in train_sizes:
				for method in methods:
					filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "f1_macro.txt")
					results.append([filename, np.loadtxt(filename)])

		results = sorted(results, key=operator.itemgetter(1))
		info = results[len(results) - 1][0].split("/")
		print info[1], info[2], info[3], info[4], results[len(results) - 1][1]

def main():
	top_results_overall()
	print
	top_results_by_feature()

if __name__ == "__main__":
	main()
