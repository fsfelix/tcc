import os
import librosa
import numpy as np
import pr_util as util
import extract_syllable_duration_4 as esd

from multiprocessing import Pool, Lock, cpu_count, RLock

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()


def mag_spec(y):
    return np.abs(librosa.core.stft(y))

def generate_local_feature(file_dir, feat_name, feat_func, log_file, norm = False, **kwargs):
    output_file = file_dir + '.' + feat_name + '.txt'
    if not os.path.isfile(output_file):
        #print('Loading {}...'.format(file_dir))
        y, sr = librosa.load(file_dir)

        if y.max() != 0.0 and norm:
            y = y/y.max()

        if feat_name == 'syllable_dur_list':
            kwargs['sr'] = sr

        #print('generating {} for {}...'.format(feat_name, file_dir))
        if len(y) > 0:
            feature = feat_func(y = y, **kwargs)
            if len(feature.shape) == 1 and feature[0] == -1:
                log_file.write('{} invalid for {}\n'.format(feat_name, file_dir))
                #print("Não encontramos {}".format(file_dir))
        else:
            log_file.write('file too short: {}\n'.format(file_dir))
            feature = np.array([0])
        np.savetxt(output_file, feature)
    else:
        log_file.write('{} already exists.'.format(output_file))
        #print('{} already exists.'.format(output_file))


def generate_local_feature_par(file_dir, feat_name, feat_func, norm = False, **kwargs):
    output_file = file_dir + '.' + feat_name + '.txt'
    if not os.path.isfile(output_file):
        print('Loading {}...'.format(file_dir))
        y, sr = librosa.load(file_dir)

        if norm:
            y = y/y.max()

        kwargs['sr'] = sr
        print('generating {} for {}...'.format(feat_name, file_dir))
        if len(y) > 0:
            feature = feat_func(y = y, **kwargs)
            if len(feature.shape) == 1 and feature[0] == -1:
                print("Não encontramos {}".format(file_dir))
        else:
            feature = np.array([0])
        np.savetxt(output_file, feature)
    else:
       print('{} already exists.'.format(output_file))

def extract_feat_par(data_dir):
    kwargs = {}
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            if util.is_audio(file):
                file_dir = subdir + '/' + file

                # y, sr = librosa.load(file_dir)

                kwargs = {}

                generate_local_feature_par(file_dir, 'rmse', librosa.feature.rmse, **kwargs)

                # generate_local_feature(file_dir, 'stft', mag_spec, log_file, **kwargs)

                generate_local_feature_par(file_dir, 'mfcc', librosa.feature.mfcc, **kwargs)

                generate_local_feature_par(file_dir, 'mfcc_norm', librosa.feature.mfcc, log_file, norm = True, **kwargs)

                generate_local_feature_par(file_dir, 'spec_cent', librosa.feature.spectral_centroid, **kwargs)

                generate_local_feature_par(file_dir, 'spec_band', librosa.feature.spectral_bandwidth, **kwargs)

                generate_local_feature_par(file_dir, 'spec_roll', librosa.feature.spectral_rolloff, **kwargs)

                generate_local_feature_par(file_dir, 'zcr', librosa.feature.zero_crossing_rate, **kwargs)

                kwargs = {'min_dur' : 0.01, 'max_dur' : 3}

                generate_local_feature_par(file_dir, 'syllable_dur', esd.get_syllable_durations, **kwargs)

                generate_local_feature_par(file_dir, 'syllable_dur_list', esd.get_syllable_durations_list, **kwargs)

def extract_feat(data_dirs, log_file):
    kwargs = {}

    n_dirs = len(data_dirs)
    n_files = int(util.num_files(data_dirs, 'song')*len(util.VERSIONS))

    i = 0
    printProgressBar(i, n_files, prefix = 'Progress:', suffix = 'Complete', length = 5)

    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if util.is_audio(file):
                    file_dir = subdir + '/' + file

                    # y, sr = librosa.load(file_dir)
                    # print(file_dir)
                    kwargs = {}

                    generate_local_feature(file_dir, 'rmse', librosa.feature.rmse, log_file, **kwargs)

                    # generate_local_feature(file_dir, 'stft', mag_spec, log_file, **kwargs)

                    generate_local_feature(file_dir, 'mfcc', librosa.feature.mfcc, log_file, **kwargs)

                    generate_local_feature(file_dir, 'mfcc_norm', librosa.feature.mfcc, log_file, norm = True, **kwargs)

                    generate_local_feature(file_dir, 'spec_cent', librosa.feature.spectral_centroid, log_file, **kwargs)

                    generate_local_feature(file_dir, 'spec_band', librosa.feature.spectral_bandwidth, log_file, **kwargs)

                    generate_local_feature(file_dir, 'spec_roll', librosa.feature.spectral_rolloff, log_file, **kwargs)

                    generate_local_feature(file_dir, 'zcr', librosa.feature.zero_crossing_rate, log_file, **kwargs)

                    kwargs = {'min_dur' : 0.01, 'max_dur' : 3}

                    # generate_local_feature(file_dir, 'syllable_dur', esd.get_syllable_durations, log_file, **kwargs)

                    generate_local_feature(file_dir, 'syllable_dur_list', esd.get_syllable_durations_list, log_file, **kwargs)
                    i += 1
                    printProgressBar(i, n_files, prefix = 'Progress:', suffix = 'Complete', length = 5)

def main():
    #pool = Pool(processes = 4)
    #pool.map(extract_feat, util.DATA_DIR_POOL)

    # Not parallel
    log_file = open(util.LOG_DIR + '/log_extract_features_' + util.date_string() + '.txt', 'w+')
    most20 = util.return_n_most_frequent_species(25, 'song')
    extract_feat(most20, log_file)
    log_file.close()

    # alldirs = util.return_n_most_frequent_species(100, 'song')
    # pool = Pool(cpu_count())
    # pool.map(extract_feat_par, alldirs)

if __name__ == '__main__':
    main()
