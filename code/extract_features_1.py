import os
import librosa
import numpy as np
import pr_util as util
import extract_syllable_duration_4 as esd

from multiprocessing import Pool

# Como rodar: extract_feat(util.DATA_DIR_FULL)

def mag_spec(y):
    return np.abs(librosa.core.stft(y))

def generate_local_feature(file_dir, feat_name, feat_func, **kwargs):
    output_file = file_dir + '.' + feat_name + '.txt'
    if not os.path.isfile(output_file):
        print('Loading {}...'.format(file_dir))
        y, sr = librosa.load(file_dir)
        print('generating {} for {}...'.format(feat_name, file_dir))
        if len(y) > 0:
            feature = feat_func(y = y, **kwargs)
        else:
            feature = np.array([0])
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

                    #y, sr = librosa.load(file_dir)

                    kwargs = {}

                    #generate_local_feature(file_dir, 'rmse', librosa.feature.rmse, **kwargs)

                    generate_local_feature(file_dir, 'stft', mag_spec, **kwargs)

                    #generate_local_feature(file_dir, 'mfcc', librosa.feature.mfcc, **kwargs)

                    #generate_local_feature(file_dir, 'spec_cent', librosa.feature.spectral_centroid, **kwargs)

                    #generate_local_feature(file_dir, 'spec_band', librosa.feature.spectral_bandwidth, **kwargs)

                    #generate_local_feature(file_dir, 'spec_roll', librosa.feature.spectral_rolloff, **kwargs)

                    # generate_local_feature(file_dir, 'zcr', librosa.feature.zero_crossing_rate, **kwargs)


                    # kwargs = {'sr':  'min_dur' : 0.01, 'max_dur' : 3}
                    # generate_local_feature(file_dir, 'syllable_dur', esd.get_syllable_durations, **kwargs)

                    # generate_local_feature(file_dir, 'syllable_dur_list', esd.get_syllable_durations_list, **kwargs)

#pool = Pool(processes = 4)
#pool.map(extract_feat, util.DATA_DIR_POOL)

most3 = ['/var/tmp/ff/tcc/dataset/pr_article/S_A_C_Base_Parte-2/Gnorimopsar chopi/', '/var/tmp/ff/tcc/dataset/pr_article/S_A_C_Base_Parte-4/Sittasomus griseicapillus/', '/var/tmp/ff/tcc/dataset/pr_article/S_A_C_Base_Parte-2/Lathrotriccus euleri/']

extract_feat(most3)
