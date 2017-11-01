import os
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from random import shuffle
from sklearn import svm, neighbors
from sklearn.model_selection import cross_val_score


NAME_SPECIES_NUM_DIR = ['Aegolius harrisii 1',
                        'Amazilia versicolor 1',
                        'Anthus lutescens 1',
                        'Attila rufus 1',
                        'Automolus leucophthalmus 1',
                        'Basileuterus leucoblepharus 1',
                        'Batara cinerea 1',
                        'Brotogeris tirica 1',
                        'Campephilus robustus 1',
                        'Camptostoma obsoletum 1',
                        'Campylorhamphus falcularius 1',
                        'Certhiaxis cinnamomeus 1',
                        'Chiroxiphia caudata 1',
                        'Chlorophanes spiza 1',
                        'Cichlocolaptes leucophrus 1',
                        'Clibanornis dendrocolaptoides 1',
                        'Cnemotriccus fuscatus 1',
                        'Colaptes campestris 1',
                        'Amazona Pretrei 1',
                        'Colonia colonus 2',
                        'Cranioleuca obsoleta 2',
                        'Crypturellus noctivagus 2',
                        'Culicivora caudacuta 2',
                        'Cyanocorax caeruleus 2',
                        'Drymophila malura 2',
                        'Dysithamnus mentalis 2',
                        'Emberizoides ypiranganus 2',
                        'Gnorimopsar chopi 2',
                        'Hemitriccus kaempferi 2',
                        'Hemitriccus orbitatus 2',
                        'Hypoedaleus guttatus 2',
                        'Lathrotriccus euleri 2',
                        'Leucochloris albicollis 2',
                        'Leucopternis polionotus 2',
                        'Mackenziaena leachii 2',
                        'Malacoptila striata 2',
                        'Mimus saturninus 2',
                        'Muscipipra vetula 2',
                        'Myiobius barbatus 2',
                        'Myiodynastes maculatus 2',
                        'Myiodynastes maculatus 3',
                        'Myiophobus fasciatus 3',
                        'Myrmeciza squamosa 3',
                        'Orthogonys chloricterus 3',
                        'Philydor atricapillus 3',
                        'Phleocryptes melanops 3',
                        'Phyllomyias griseocapilla 3',
                        'Phylloscartes kronei 3',
                        'Picumnus temminckii 3',
                        'Piprites chloris 3',
                        'Piprites pileata 3',
                        'Polioptila dumicola 3',
                        'Poospiza nigrorufa 3',
                        'Procnias nudicollis 3',
                        'Pseudoleistes guirahuro 3',
                        'Pyriglena leucoptera 3',
                        'Ramphastos dicolorus 3',
                        'Ramphocelus bresilius 3',
                        'Ramphodon naevius 3',
                        'Saltator similis 3',
                        'Schiffornis virescens 4',
                        'Scytalopus iraiensis 4',
                        'Sittasomus griseicapillus 4',
                        'Streptoprocne biscutata 4',
                        'Stymphalornis acutirostris 4',
                        'Synallaxis spixi 4',
                        'Tangara desmaresti 4',
                        'Tangara peruviana 4',
                        'Thamnophilus ruficapillus 4',
                        'Theristicus caudatus 4',
                        'Thraupis palmarum 4',
                        'Thryothorus longirostris 4',
                        'Trichothraupis melanops 4',
                        'Trogon surrucura 4',
                        'Vanellus chilensis 4',
                        'Xenops minutus 4',
                        'Xiphorhynchus fuscus 4']

#DATA_DIR_BASE = '/var/tmp/ff/pr_article/S_A_C_Base_Parte'
DATA_DIR_BASE = '/var/tmp/ff/tcc/dataset/pr_article/S_A_C_Base_Parte'

DATA_DIR_ORIGINAL = '/var/tmp/ff/tcc/dataset/pr_article/original/S_A_C_Base_Parte'

DATA_DIR_FULL = [DATA_DIR_BASE + '-1/',
                 DATA_DIR_BASE + '-2/',
                 DATA_DIR_BASE + '-3/',
                 DATA_DIR_BASE + '-4/']

DATA_DIR_POOL = [[DATA_DIR_BASE + '-1/'],
                 [DATA_DIR_BASE + '-2/'],
                 [DATA_DIR_BASE + '-3/'],
                 [DATA_DIR_BASE + '-4/']]

#DATA_DIR_PULSE_BASE = '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Pulsos_Parte'

DATA_DIR_PULSE_BASE = '/var/tmp/ff/tcc/dataset/pr_article/S_A_C_Base_Pulsos_Parte'

DATA_DIR_PULSE_FULL = [DATA_DIR_PULSE_BASE + '-1/',
                       DATA_DIR_PULSE_BASE + '-2/',
                       DATA_DIR_PULSE_BASE + '-3/',
                       DATA_DIR_PULSE_BASE + '-4/']

DATA_DIR_PULSE_POOL = [[DATA_DIR_PULSE_BASE + '-1/'],
                       [DATA_DIR_PULSE_BASE + '-2/'],
                       [DATA_DIR_PULSE_BASE + '-3/'],
                       [DATA_DIR_PULSE_BASE + '-4/']]


EXPERIMENTS_DIR = '/var/tmp/ff/tcc/experiments'

FEATURES = ['rmse', 'mfcc', 'spec_band', 'spec_cent', 'spec_roll', 'syllable_dur', 'syllable_dur_list', 'zcr']

CLASSIFIERS = ['kNN', 'NB', 'SVM']

GLOBAL_FUNCTIONS = [np.mean, np.std, np.max, np.min]

VERSIONS = [None, 'filtered1', 'filtered2', 'filtered3', 'filtered4']
#VERSIONS = [None] # no need for this anymore.
VERSIONS_EXPERIMENTS = [None, 'filtered1', 'filtered4']

def is_audio(file_name):
    file_extension = file_name.split('.')[-1]
    file_extension = file_extension.lower()
    return file_extension == 'mp3' or file_extension == 'wav' or file_extension == 'flac' or file_extension == 'aiff' or file_extension == 'aac'

def is_not_wav(file_name):
    file_extension = file_name.split('.')[-1]
    file_extension = file_extension.lower()
    return file_extension == 'mp3' or file_extension == 'flac' or file_extension == 'aiff' or file_extension == 'aac'

def num_files(data_dirs, song_or_call, num_versions = 1):
    # num_versions indicates how many filtered versions we
    # have for each original audio file

    num_versions = len(VERSIONS)
    #num_versions = count_versions(data_dirs[0], song_or_call)
    num_file = 0
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                type_of_rec = subdir.split('/')[-1]
                if is_audio(file) and type_of_rec == song_or_call:
                #if type_of_rec == song_or_call:
                    #print(subdir + '/' + file)
                    num_file += 1
    return int(num_file/num_versions)

def generate_filtered_dirs(data_dir, num_filters = 3):
    recordings = [data_dir]
    for i in range(num_filters):
        recordings.append(data_dir + '.filtered' + str(i + 1) + '.wav')
    return recordings

def return_random_audio(data_dirs):
    recordings = []
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                type_of_rec = subdir.split('/')[-1] # Is it a call or a song?
                filt_count  = file.count('filtered')
                if type_of_rec == 'song' and is_audio(file) and filt_count == 0:
                    recordings.append(subdir + '/' + file)
    rec_choosen = recordings[np.random.randint(len(recordings))]
    return generate_filtered_dirs(rec_choosen)

def plot_scatter(x, y, labels, xlabel, ylabel):
    # plot scatter graph with 2 features

    fig, ax = plt.subplots()
    markers = ['2', '.', '>', '*', '<', ',', '1', '8']
    colors = ['b','g','r','c','m','y','k','w']

    for label in labels:
        ax.scatter(x[labels == label], y[labels == label], marker = markers[label], c = colors[label], s = 15)

    plt.title("Scatter Plot (n_species = %i)" % (max(labels) + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def samples_to_time(n_samples, sr):
    return n_samples/sr

def time_to_samples(time_sec, sr):
    return int(time_sec * sr)

def audio_to_txt(file_dir):
    y, sr = librosa.load(file_dir)
    y = np.append(y, sr)
    print('Writing {}...'.format(file_dir_txt(file_dir)))
    np.savetxt(file_dir_txt(file_dir), y)

def txt_to_audio(file_dir):
    y = np.loadtxt(file_dir)
    sr = y[-1]
    return y[:-1], sr

def dirs_to_pulse_dirs(data_dirs):
    dirs = []
    for data_dir in data_dirs:
        dirs.append(data_dir.replace('Base', 'Base_Pulsos'))
    return dirs

def count_versions(data_dir, song_or_call):
    num_versions = 0
    for subdir, dirs, files in os.walk(data_dir):
        for file in files:
            type_of_rec = subdir.split('/')[-1]
            if is_audio(file) and type_of_rec == song_or_call:
                spl = file.split('.')
                for s in spl:
                    if s.count('filtered') != 0:
                        current = int(s[8:])
                        if current > num_versions:
                            num_versions = current + 1
    return num_versions if num_versions != 0 else 1
    #return int(num_file/num_versions)

def number_of_dir_and_name(name_dir):
    return name_dir[:-2], name_dir[-1]

def create_list_with_dir_and_number(song_or_call):
    dir_number = []
    for name_dir in NAME_SPECIES_NUM_DIR:
        spc, num_dir = number_of_dir_and_name(name_dir)
        spc_dir = DATA_DIR_BASE + '-' +  num_dir + '/' + spc + '/'
        # spc_dir = DATA_DIR_ORIGINAL + '-' +  num_dir + '/' + spc + '/'
        num_dir = num_files([spc_dir], song_or_call)
        #print(spc_dir + ' ' +  str(num_dir))
        dir_number.append([spc_dir, num_dir])
    dir_number.sort(key = lambda x:x[1])
    return dir_number

def choose_species(num_spc, num_min, num_max, song_or_call):
    spcs_filtered = []
    spcs = create_list_with_dir_and_number(song_or_call)

    for spc in spcs:
        if spc[1] >= num_min and spc[1] <= num_max:
            spcs_filtered.append(spc[0])

    shuffle(spcs_filtered)
    return spcs_filtered[:num_spc]
