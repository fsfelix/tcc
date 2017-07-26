import glob, os, sys
import numpy as np

from sklearn import naive_bayes
from sklearn import tree
from sklearn import neighbors
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

AUDIO_DIR = "gtzan"
LOCAL_FEATURE_DIR = "local_features"
GLOBAL_FEATURE_DIR = "global_features"
RESULTS_DIR = "results"

genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]

n_songs = 100

features = ["chroma_stft", "chroma_cqt", "chroma_cens", "melspectrogram",
	"mfcc", "rmse", "spectral_centroid", "spectral_bandwidth", "spectral_contrast",
	"spectral_rolloff", "poly_features", "tonnetz", "zero_crossing_rate", "tempogram"]

classifiers = [
	[naive_bayes.GaussianNB(), "GNB", None], # GNB does not have params to be optimized
	[tree.DecisionTreeClassifier(), "DTC", {
		'criterion': ['gini', 'entropy'],
		'class_weight': [None, 'balanced'],
		'splitter': ['best', 'random'],
		'max_features': [None, 'auto', 'sqrt', 'log2'],
		'min_samples_split': [2, 3, 5, 7, 11, 13, 17],
		'min_samples_leaf': [1, 2, 3, 5, 7, 11, 13, 17],
	}],
	[neighbors.KNeighborsClassifier(), "KNC", {
		'n_neighbors': range(1, 30),
		'weights': ['uniform', 'distance'],
		'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
		'n_jobs': [4]
	}],
	[svm.SVC(), "SVC", {
		'kernel': ['rbf'],
		'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1],
		'C': range(1, 1000, 100)
	}]
]

def prepare_data(feature):
	data = []
	target = []

	for genre in genres:
		global_filename = os.path.join(GLOBAL_FEATURE_DIR, genre, genre + "." + feature)
		global_file = np.loadtxt(global_filename)
		for i in range(global_file.shape[0]):
			data.append(global_file[i])
			target.append(genre)

	return np.array(data), np.array(target)

def prepare_splits_and_call_function(feature, classifier, function):
	data, target = prepare_data(feature)

	print("50/50 split")
	X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.5, random_state=0)
	function(feature, "50/50", classifier, X_train, X_test, y_train, y_test)

	print("60/40 split")
	X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.4, random_state=0)
	function(feature, "60/40", classifier, X_train, X_test, y_train, y_test)

	print("70/30 split")
	X_train, X_test, y_train, y_test = model_selection.train_test_split(data, target, test_size=0.3, random_state=0)
	function(feature, "70/30", classifier, X_train, X_test, y_train, y_test)

def classify(classifier, X_train, y_train, X_test):
	fit = classifier.fit(X_train, y_train)
	return classifier.predict(X_test)

def grid_search(classifier, X_train, y_train, X_test):
	clf = GridSearchCV(classifier[0], classifier[2], cv=5, scoring='f1_macro')
	clf.fit(X_train, y_train)
	return clf.predict(X_test)

def transform_data(X_train, X_test, n_components, function, *args):
	f = function(n_components=n_components)
	fit = f.fit(*args)
	X_train_t = fit.transform(X_train)
	X_test_t = fit.transform(X_test)
	return X_train_t, X_test_t

def check_all_n_components(classifier, X_train, X_test, y_train, y_test, function, *args):
	max_hit_rate = 0
	max_n_components = None

	for n_components in range(1, X_train.shape[1] + 1):
		X_train_t, X_test_t = transform_data(X_train, X_test, n_components, function, *args)
		y_pred = classify(classifier, X_train_t, y_train, X_test_t)
		hit_rate = 100.0 * (y_test == y_pred).sum() / y_test.shape[0]

		if hit_rate > max_hit_rate:
			max_hit_rate = hit_rate
			max_n_components = n_components
			# TODO: check out f.explained_variance_ratio_

	return max_hit_rate, max_n_components

def save_data_to_txt(filename, data):
	if not os.path.isfile(filename):
		np.savetxt(filename, data)

def get_all_features():
	f = []
	for feature in features:
		f.append(feature)
		f.append(feature + "_beat")
	return f

def get_classifier_names():
	names = []
	for classifier in classifiers:
		names.append(classifier[1])
	return names
