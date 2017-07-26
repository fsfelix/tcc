import os
import numpy as np
import matplotlib.pyplot as plt
import utility_functions as util

train_sizes = ["50", "60", "70"]
methods = ["all", "pca", "lda"]
features = util.get_all_features()
classifiers = util.get_classifier_names()

def generate_data(results, group):
	i_n = len(results.itervalues().next())
	j_n = len(group)
	data = np.empty((i_n, j_n))
	for i in range(i_n):
		for j in range(j_n):
			data[i][j] = results[group[j]][i]
	return data

def save_figure(data, ticks, rotation, filename):
	print "saving " + filename
	plt.figure()
	plt.boxplot(data)
	tick_marks = np.arange(len(ticks) + 1)
	plt.xticks(tick_marks, [''] + ticks, rotation=rotation)
	plt.tight_layout()
	plt.savefig(filename)

def plot_fixed_feature():
	fixed_feature_results = {}
	for feature in features:
		results = []
		for classifier in classifiers:
			for train_size in train_sizes:
				for method in methods:
					filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "f1_macro.txt")
					results.append(np.loadtxt(filename))
		fixed_feature_results[feature] = results

	data = generate_data(fixed_feature_results, features)
	save_figure(data, features, 90, "features.png")

def plot_fixed_classifier():
	fixed_classifier_results = {}
	for classifier in classifiers:
		results = []
		for feature in features:
			for train_size in train_sizes:
				for method in methods:
					filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "f1_macro.txt")
					results.append(np.loadtxt(filename))
		fixed_classifier_results[classifier] = results

	data = generate_data(fixed_classifier_results, classifiers)
	save_figure(data, classifiers, 45, "classifiers.png")

def plot_fixed_train_size():
	fixed_train_size_results = {}
	for train_size in train_sizes:
		results = []
		for feature in features:
			for classifier in classifiers:
				for method in methods:
					filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "f1_macro.txt")
					results.append(np.loadtxt(filename))
		fixed_train_size_results[train_size] = results

	data = generate_data(fixed_train_size_results, train_sizes)
	save_figure(data, train_sizes, 45, "train_sizes.png")

def plot_fixed_method():
	fixed_method_results = {}
	for method in methods:
		results = []
		for feature in features:
			for classifier in classifiers:
				for train_size in train_sizes:
					filename = os.path.join(util.RESULTS_DIR, feature, train_size, classifier, method, "f1_macro.txt")
					results.append(np.loadtxt(filename))
		fixed_method_results[method] = results

	data = generate_data(fixed_method_results, methods)
	save_figure(data, methods, 45, "methods.png")

def main():
	plot_fixed_feature()
	plot_fixed_classifier()
	plot_fixed_train_size()
	plot_fixed_method()

if __name__ == "__main__":
	main()
