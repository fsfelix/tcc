import os
import librosa
import pr_util as util

from my_filters import *

#audiofiles = [x for x in os.listdir() if (x.endswith('.wav') or x.endswith('.mp3') or x.endswith('.flac') or x.endswith('.aiff')) or x.endswith('.aac')]

#for file in audiofiles:
  # y, sr = librosa.load(file, sr=44100)
  # y_filtered = my_filter(y, sr)
  # path = 'filtered/'+file

  # librosa.output.write_wav(path, y_filtered, sr)
  # print('Filtered ' + file)


# data_dirs = ['/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-1/Batara cinerea/',
#              '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-1/Camptostoma obsoletum/',
#              '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-3/Myiodynastes maculatus/']

data_dirs = ['/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-1/',
                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-2/',
                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-3/',
                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-4/']


for data_dir in data_dirs:
  for subdir, dirs, files in os.walk(data_dir):
    for file in files:
      if util.is_audio(file) and file.count('filtered') == 0:
        file_dir = subdir + '/' + file
        print(file_dir)
        y, sr = librosa.load(file_dir, sr=44100)
        y_filtered = my_filter2(y, sr)
        path = file_dir + '.filtered2.wav'
        print("arquivo filtrado: {}".format(path))
        librosa.output.write_wav(path, y_filtered, sr)
        #print('Filtered ' + file)
