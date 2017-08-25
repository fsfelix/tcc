import os
import librosa
import numpy as np
import pr_util as util

# Como rodar: extract_feat(util.DATA_DIR_FULL)

def generate_local_feature(file_dir, feat_name, feat_func, y, **kwargs):
    output_file = file_dir + '.' + feat_name + '.txt'
    if not os.path.isfile(output_file):
        print('generating {} for {}...'.format(feat_name, file_dir))
        feature = feat_func(y = y, **kwargs)
        np.savetxt(output_file, feature)
    else:
       print('{} already exists.'.format(output_file))

def extract_feat(data_dirs):
    kwargs = {}

    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if util.is_audio(file):
                    file_dir = subdir + '/' + file
                    #print('Loading {}...'.format(file_dir))
                    y, sr = librosa.load(file_dir)
                    kwargs = {'sr' : sr}

                    # generate_local_feature(file_dir, 'rmse', librosa.feature.rmse, y)

                    # generate_local_feature(file_dir, 'stft', librosa.core.stft, y)

                    # generate_local_feature(file_dir, 'mfcc', librosa.feature.mfcc, y, **kwargs)

                    generate_local_feature(file_dir, 'spec_cent', librosa.feature.spectral_centroid, y, **kwargs)

                    # generate_local_feature(file_dir, 'spec_band', librosa.feature.spectral_bandwidth, y, **kwargs)

                    # generate_local_feature(file_dir, 'spec_roll', librosa.feature.spectral_rolloff, y, **kwargs)

#extract_feat(util.DATA_DIR_FULL)
