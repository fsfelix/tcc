import os
import numpy as np
import pr_util as util

def update_labels_dict(bird_specie, labels_dict, labels, n_label):
    if not bird_specie in labels_dict.keys():
        n_label += 1
        labels_dict[bird_specie] = n_label
        labels.append(n_label)
    else:
        labels.append(labels_dict[bird_specie])
    return n_label

def generate_global_multi_features(n_global_feat, feat_names, data_dirs, song_or_call, functions, version = None):
    labels_dict = {}
    labels = []
    n_label = -1
    n_files = util.num_files(data_dirs, song_or_call)
    i = 0
    j = 0
    current_feature = 0
    n_feat = len(feat_names)
    data = np.empty((n_files, n_global_feat * n_feat))

    for feat_name in feat_names:
        i = 0
        for data_dir in data_dirs:
            for subdir, dirs, files in os.walk(data_dir):
                for file in files:
                    type_of_rec = subdir.split('/')[-1] # Is it a call or a song?
                    if type_of_rec == song_or_call and file.split('.')[-2] == feat_name:
                        filt_count = file.count('filtered')
                        if (filt_count == 1 and file.split('.')[-4] == version) or (filt_count == 0 and version == None):
                            bird_specie = subdir.split('/')[-2].title()
                            # if not bird_specie in labels_dict.keys():
                            #     n_label += 1
                            #     labels_dict[bird_specie] = n_label
                            #     labels.append(n_label)
                            # else:
                            #     labels.append(labels_dict[bird_specie])
                            if current_feature == 0: # we only create labels_dict and labels one time.
                                n_label = update_labels_dict(bird_specie, labels_dict, labels, n_label)
                            feature_path = subdir + '/' + file
                            feature = np.loadtxt(feature_path)
                            for function in functions: # Iterate through all functions
                                #print(feature_path)
                                if feature.size > 0:
                                    data[i][(n_global_feat * current_feature) + j] = function(feature)
                                else:
                                    data[i][(n_global_feat * current_feature) + j] = 0
                                j += 1
                            i += 1
                            j  = 0
        current_feature += 1

    labels = np.array(labels)
    return labels_dict, labels, data

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
    #data = np.empty((n_files, n_global_feat))
    data = []
    print("number of files loaded: {}".format(n_files))
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                type_of_rec = subdir.split('/')[-1] # Is it a call or a song?
                if type_of_rec == song_or_call and file.split('.')[-2] == feat_name:
                    filt_count = file.count('filtered')
                    if (filt_count == 1 and file.split('.')[-4] == version) or (filt_count == 0 and version == None):
                        #print(file)
                        bird_specie = subdir.split('/')[-2].title()
                        if not bird_specie in labels_dict.keys():
                            n_label += 1
                            labels_dict[bird_specie] = n_label
                            labels.append(n_label)
                        else:
                            labels.append(labels_dict[bird_specie])
                        feature_path = subdir + '/' + file
                        feature = np.loadtxt(feature_path)
                        data.append(create_global_feat_data(feature, functions))
    data = np.array(data)
    labels = np.array(labels)
    return labels_dict, labels, data

def create_global_feat_data(feat, functions):
    data = []
    global_feat = [0]*len(functions)

    if len(feat.shape) > 1:
        for i in range(len(functions)):
            f = functions[i]
            kwargs = {'axis' : 1}
            global_feat[i] = [f(feat, **kwargs)]
        data = np.array(global_feat)
        data = data.T.reshape((1, data.size))
        return data[0]
    else:
        for i in range(len(functions)):
            f = functions[i]
            global_feat[i] = f(feat)
        data = np.array(global_feat)
        return data
