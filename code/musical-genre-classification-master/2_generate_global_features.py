import os
import numpy as np
import utility_functions as util

def generate_global(feature, n_features):
	print feature
	for genre in util.genres:
		print "\t" + genre

		global_filename = os.path.join(util.GLOBAL_FEATURE_DIR, genre, genre + "." + feature)

		if not os.path.isfile(global_filename):
			data = np.empty((util.n_songs, n_features * 4))

			for i in range(util.n_songs):
				feature_filename = os.path.join(util.LOCAL_FEATURE_DIR, genre, genre + "." + str(i).zfill(5) + "." + feature)
				feature_file = np.loadtxt(feature_filename)

				j = 0
				for k in range(n_features):
					data[i][j] = np.mean(feature_file[k])
					data[i][j + 1] = np.std(feature_file[k])
					data[i][j + 2] = np.min(feature_file[k])
					data[i][j + 3] = np.max(feature_file[k])
					j += 4

			np.savetxt(global_filename, data)

def generate_global_features(feature):
	example_feature_filename = os.path.join(util.LOCAL_FEATURE_DIR, "blues", "blues.00000." + feature)
	example_feature = np.loadtxt(example_feature_filename)

	if example_feature.ndim == 1:
		generate_global(feature, 1)

	if example_feature.ndim == 2:
		n_features = example_feature.shape[0]
		generate_global(feature, n_features)

def main():
	for feature in util.features:
		generate_global_features(feature)
		generate_global_features(feature + "_beat")

if __name__ == "__main__":
	main()
