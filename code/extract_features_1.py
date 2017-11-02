import os
import librosa
import numpy as np
import pr_util as util
import extract_syllable_duration_4 as esd

from multiprocessing import Pool

def mag_spec(y):
    return np.abs(librosa.core.stft(y))

def generate_local_feature(file_dir, feat_name, feat_func, log_file, **kwargs):
    output_file = file_dir + '.' + feat_name + '.txt'
    if not os.path.isfile(output_file):
        print('Loading {}...'.format(file_dir))
        y, sr = librosa.load(file_dir)
        kwargs['sr'] = sr
        print('generating {} for {}...'.format(feat_name, file_dir))
        if len(y) > 0:
            feature = feat_func(y = y, **kwargs)
            if feature[0] == -1:
                log_file.write('{} invalid for {}\n'.format(feat_name, file_dir))
                print("Não encontramos {}".format(file_dir))
        else:
            log_file.write('file too short: {}\n'.format(file_dir))
            feature = np.array([0])
        np.savetxt(output_file, feature)
    else:
       print('{} already exists.'.format(output_file))

def extract_feat(data_dirs, log_file):
    kwargs = {}

    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if util.is_audio(file):
                    file_dir = subdir + '/' + file

                    # y, sr = librosa.load(file_dir)

                    kwargs = {}

                    #generate_local_feature(file_dir, 'rmse', librosa.feature.rmse, log_file, **kwargs)

                    # generate_local_feature(file_dir, 'stft', mag_spec, log_file, **kwargs)

                    #generate_local_feature(file_dir, 'mfcc', librosa.feature.mfcc, log_file, **kwargs)

                    #generate_local_feature(file_dir, 'spec_cent', librosa.feature.spectral_centroid, log_file, **kwargs)

                    #generate_local_feature(file_dir, 'spec_band', librosa.feature.spectral_bandwidth, log_file, **kwargs)

                    #generate_local_feature(file_dir, 'spec_roll', librosa.feature.spectral_rolloff, log_file, **kwargs)

                    # generate_local_feature(file_dir, 'zcr', librosa.feature.zero_crossing_rate, log_file, **kwargs)
                    kwargs = {'min_dur' : 0.01, 'max_dur' : 3}

                    # generate_local_feature(file_dir, 'syllable_dur', esd.get_syllable_durations, log_file, **kwargs)

                    generate_local_feature(file_dir, 'syllable_dur_list', esd.get_syllable_durations_list, log_file, **kwargs)

def main():
    #pool = Pool(processes = 4)
    #pool.map(extract_feat, util.DATA_DIR_POOL)

    log_file = open(util.LOG_DIR + '/log_extract_features_' + util.date_string() + '.txt', 'w+')
    most20 = util.return_n_most_frequent_species(20, 'song')
    extract_feat(most20, log_file)
    log_file.close()

if __name__ == '__main__':
    main()
