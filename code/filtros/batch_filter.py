import sys
sys.path.insert(0, '../')
import os
import librosa
import pr_util as util

from my_filters import *

from multiprocessing import Pool, Lock, cpu_count, RLock

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

#data_dirs = ['/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-1/',
#                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-2/',
#                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-3/',
#                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-4/']

#data_dirs = util.DATA_DIR_FULL


#for data_dir in util.DATA_DIR_FULL:

def my_filter_run(data_dir):
  filtered_version = '.filtered5.wav'
  for subdir, dirs, files in os.walk(data_dir):
    for file in files:
      if util.is_audio(file) and file.count('filtered') == 0:
        file_dir = subdir + '/' + file
        filtered_dir = file_dir + filtered_version
        if not os.path.isfile(filtered_dir):
          print(file_dir)
          y, sr = librosa.load(file_dir)
          #y_filtered = my_filter3(y, sr)
          #y_filtered = my_filter4(y, util.time_to_samples(0.5, sr))
          y_filtered = my_filter5(y, util.time_to_samples(0.5, sr))
          print("arquivo filtrado: {}".format(filtered_dir))
          librosa.output.write_wav(filtered_dir, y_filtered, sr)
        else:
          print('{} already exists.'.format(filtered_dir))

def main():
  pool = Pool(cpu_count())
  DIRS = util.return_n_most_frequent_species(100, 'song')
  pool.map(my_filter_run, DIRS)

if __name__ == '__main__':
    main()
