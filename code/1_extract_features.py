import os
import librosa
import numpy as np
import pr_util as util

def generate_local_feature(file_dir, feat_name, feat_func, y, **kwargs):
    output_file = file_dir + '.' + feat_name + '.txt'
    if not os.path.isfile(output_file):
        print('generating {} for {}...'.format(feat_name, file_dir))
        feature = feat_func(y = y, **kwargs)
        np.savetxt(output_file, feature)
    else:
       print('{} already exists.'.format(output_file))

def main():
    kwargs = {}

    for data_dir in util.DATA_DIR_FULL:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if util.is_audio(file):
                    file_dir = subdir + '/' + file
                    #print('Loading {}...'.format(file_dir))
                    y, sr = librosa.load(file_dir)

                    #generate_local_feature(file_dir, 'rmse', librosa.feature.rmse, y)

                    #generate_local_feature(file_dir, 'stft', librosa.core.stft, y)

                    generate_local_feature(file_dir, 'mfcc', librosa.feature.mfcc, y, **kwargs)


if __name__ == "__main__":
    main()
