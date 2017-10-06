import os
import time
import librosa
import numpy as np
import pr_util as util

def file_dir_wav(file_dir):
    spl = file_dir.split('.')
    spl[-1] = 'wav'
    new = ''
    for s in spl:
        new += s + '.'
    return new[:-1]

def file_dir_txt(file_dir):
    spl = file_dir.split('.')
    spl[-1] = 'txt'
    new = ''
    for s in spl:
        new += s + '.'
    return new[:-1]

def audio_to_txt(file_dir):
    y, sr = librosa.load(file_dir)
    y = np.append(y, sr)
    print('Writing {}...'.format(file_dir_txt(file_dir)))
    np.savetxt(file_dir_txt(file_dir), y)

def txt_to_audio(file_dir):
    y = np.loadtxt(file_dir)
    sr = y[-1]
    return y[:-1], sr

def all_to_wav(data_dirs):
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if util.is_not_wav(file):
                    file_dir = subdir + '/' + file
                    print('Loading {}...'.format(file_dir))
                    y, sr = librosa.load(file_dir)
                    file_dir = file_dir_wav(file_dir)
                    print('Writing {}...'.format(file_dir))
                    librosa.output.write_wav(file_dir, y, sr)

def all_to_txt(data_dirs):
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                if util.is_audio(file):
                    file_dir = subdir + '/' + file
                    print('Loading {}...'.format(file_dir))
                    audio_to_txt(file_dir)
                    #librosa.output.write_wav(file_dir, y, sr)

all_to_txt(['/Users/felipefelix/USP/tcc/dataset/pr_article/experimentos_100/Vanellus chilensis', '/Users/felipefelix/USP/tcc/dataset/pr_article/experimentos_100/Trogon surrucura', '/Users/felipefelix/USP/tcc/dataset/pr_article/experimentos_100/Synallaxis spixi'])
