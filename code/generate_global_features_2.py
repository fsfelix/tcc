import os
import numpy as np
import pr_util as util

def generate_global_features(n_global_feat, feat_name, data_dirs, song_or_call, functions, version = None):
    # n_global_feat: number of global features
    # feat_name: feature name, must use the convection file.feat_name.txt
    # data_dirs: list with directories with birds features
    # song_or_call: what type of recording is to be used
    # functions: list with functions that will generate global features [np.max, etc...]

    labels_dict = {}
    labels  = []
    n_label = -1
    n_files = util.num_files(data_dirs, song_or_call)
    i = 0
    j = 0
    data = np.empty((n_files, n_global_feat))

    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                type_of_rec = subdir.split('/')[-1] # Is it a call or a song?
                if type_of_rec == song_or_call and file.split('.')[-2] == feat_name:
                    filt_count = file.count('filtered')
                    if (filt_count == 1 and file.split('.')[-4] == version) or (filt_count == 0 and version == None):
                        print(file)
                        bird_specie = subdir.split('/')[-2].title()
                        if not bird_specie in labels_dict.keys():
                            n_label += 1
                            labels_dict[bird_specie] = n_label
                            labels.append(n_label)
                        else:
                            labels.append(labels_dict[bird_specie])
                        feature_path = subdir + '/' + file
                        feature = np.loadtxt(feature_path)

                        for function in functions: # Iterate through all functions
                            data[i][j] = function(feature)
                            j += 1
                        i += 1
                        j  = 0

    labels = np.array(labels)
    return labels_dict, labels, data
