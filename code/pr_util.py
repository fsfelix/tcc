import numpy as np
import os

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

DATA_DIR_BASE = '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte'

DATA_DIR_FULL = ['/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-1/',
                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-2/',
                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-3/',
                 '/Users/felipefelix/USP/tcc/dataset/pr_article/S_A_C_Base_Parte-4/']

def is_audio(file_name):
    file_extension = file_name.split('.')[-1]
    file_extension = file_extension.lower()
    return file_extension == 'mp3' or file_extension == 'wav' or file_extension == 'flac' or file_extension == 'aiff' or file_extension == 'aac'

def num_files(data_dirs, song_or_call):
    num_file = 0
    for data_dir in data_dirs:
        for subdir, dirs, files in os.walk(data_dir):
            for file in files:
                type_of_rec = subdir.split('/')[-1]
                if is_audio(file) and type_of_rec == song_or_call:
                    num_file += 1
    return num_file

def choose_species(num_species):
    # Return list of directories with species randomly choosen

    species =  np.random.choice(NAME_SPECIES_NUM_DIR, num_species)
    dirs = []

    for specie in species:
        dir = DATA_DIR_BASE + '-' + specie[-1] + '/' + specie[:-2] + '/'
        print(dir)
        dirs.append(dir)

    return dirs

def plot_scatter(x, y, labels, xlabel, ylabel):
    # plot scatter graph with 2 features

    fig, ax = plt.subplots()
    markers = ['2', '.', '>', '*', '<', ',', '1', '8']
    colors = ['b','g','r','c','m','y','k','w']

    for label in labels:
        ax.scatter(x[labels == label], y[labels == label], marker = markers[label], c = colors[label], s = 75)

    plt.title("Scatter Plot (n_species = %i)" % (max(labels) + 1))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

def kNN(data, labels,k_range, cv = 5):
    for k in k_range:
        for weight in ['uniform', 'distance']:
            clf = neighbors.KNeighborsClassifier(k, weights = weight)
            scores = cross_val_score(clf, data, labels, cv = cv)
            print("{0}-Neighbors | Accuracy: {1:.2f} (+/- {2:.2f}) | Weight: {3}".format(k, scores.mean(), scores.std() * 2, weight))
