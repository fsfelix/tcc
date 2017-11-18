# Máscaras:

# Variância de energia ao longo do tempo:

# A cada 200 frames de tempo, é calculado o desvio padrão de energia ao longo
# desses 200 frames para cada linha de frequencia da matriz do espectro,
# resultando em um vetor de 1 x n linhas de frequência. Esse vetor é replicado
# 200 vezes, e o processo é repetido até compor uma matriz com as mesmas
# dimensões do espectro. Ao fim da composição, normalizamos a matriz inteira
# dividindo pelo maior valor.

# Contraste espectral:

# É usado o contraste espectral do libROSA, com 8 oitavas e primeira oitava de 0
# a 64Hz. P/ cada oitava, em cada frame de tempo é calculada a diferença linear
# entre o pico e vale de amplitude, e é retornada uma matriz com dimensões n x 9,
# onde n é o número de frames de tempo do espectro. A matriz é espandida para
# ficar com dimensões iguais ao espectro, e normalizada.

# Filtros:

# Filtro 1: as duas máscaras são multiplicadas elemento a elemento, e a matriz
# resultante é novamente normalizada. Depois, o espectro é multiplicado por essa
# matriz.

# Filtro 2: primeiro multiplicamos o espectro pela máscara de variância. Depois,
# calculamos o contraste espectral desse sinal pré-filtrado, e filtramos
# novamente com a máscara do contraste.

# Filtro 3: a máscara final de filtragem é binária: caso um bin do espectro
# possua bin correspondente da máscara de variância maior que threshold_var E bin
# correspondente da máscara do contraste maior que threshold_contrast, a máscara
# final de filtragem recebe 1. Caso contrário, 0. Esses thresholds são fixos


import librosa
import numpy as np


def normalizedVar(Y):
    var_p = np.sqrt(Y.var(1))
    var_p = var_p.reshape(var_p.shape[0],1)
    return(var_p)


def varTrustFunc(Y, numFrames=200):
    x = int(Y.shape[1]/numFrames)
    varTrust = np.ones(Y.shape)
    for i in range(0, Y.shape[1], numFrames):
        indiceStart = i
        indiceStop = indiceStart+numFrames
        Y_p = Y[:,indiceStart:indiceStop]

        var_p = normalizedVar(Y_p)
        varTrust[:,indiceStart:indiceStop] = var_p
    varTrust = varTrust / np.max(varTrust)
    return(varTrust)


def expandContrast(contrast_p, shape, n_bands, deltaF):
    contrast = np.ones(shape)
    for i in range(0, n_bands+1):
        if i == 0:
            indiceStart = 0
        else:
            indiceStart = deltaF * 2**(i-1)
        indiceStop = deltaF * 2**i - 1
        contrast[indiceStart:indiceStop,:] = contrast_p[i,:]
    contrast[indiceStop+1:,:] = contrast_p[n_bands,:]
    return(contrast)


def contrastTrustFunc(Y, sr):
    n_bands = 8
    contrast_p = librosa.feature.spectral_contrast(S=Y, sr=sr, linear=True, n_bands=n_bands, fmin=64)
    deltaF = int(round(64/(sr/(2*Y.shape[0]))))
    contrast = expandContrast(contrast_p, Y.shape, n_bands, deltaF)
    contrast /= np.max(contrast)
    return(contrast)


# In[17]:

def my_filter(y, sr):
    Y = librosa.stft(y, n_fft = 4096, hop_length = 512)
    Y_dB = librosa.amplitude_to_db(Y, ref=np.max)
    varTrust = varTrustFunc(Y_dB)
    contrast = contrastTrustFunc(np.abs(Y), sr)
    mask = np.multiply(contrast, varTrust)
    mask = mask / np.max(mask)
    mag, phase = librosa.magphase(Y)
    newmag = np.multiply(mag, mask)
    Y_rec = np.multiply(newmag, np.exp(np.multiply(phase, (1j))))
    y_rec = librosa.istft(Y_rec, hop_length=512)

    return y_rec


# In[18]:

def my_filter2(y, sr):
    Y = librosa.stft(y, n_fft = 4096, hop_length = 512)
    Y_dB = librosa.amplitude_to_db(Y, ref=np.max)
    varTrust = varTrustFunc(Y_dB)
    mask = varTrust
    mag, phase = librosa.magphase(Y)
    newmag = np.multiply(mag, mask)
    contrast = contrastTrustFunc(np.abs(newmag), sr)
    mask = contrast
    x = 0.2
    mask[mask < x] = x
    newnewmag = np.multiply(newmag, mask)
    Y_rec = np.multiply(newnewmag, np.exp(np.multiply(phase, (1j))))

    y_rec = librosa.istft(Y_rec, hop_length=512)

    return y_rec


# In[19]:

def my_filter3(y, sr):

    Y = librosa.stft(y, n_fft = 4096, hop_length = 512)
    Y_dB = librosa.amplitude_to_db(Y, ref=np.max)

    varTrust = varTrustFunc(Y_dB)
    contrast = contrastTrustFunc(np.abs(Y), sr)

    mask = np.zeros(varTrust.shape)
    mask[np.logical_and(varTrust > 0.15, contrast > 0.2)] = 1

    mag, phase = librosa.magphase(Y)
    newmag = np.multiply(mag, mask)
    Y_rec = np.multiply(newmag, np.exp(np.multiply(phase, (1j))))

    y_rec = librosa.istft(Y_rec, hop_length=512)

    return y_rec

# Segmentação antiga de energia.

def my_filter4(y, size):
    N = len(y)
    energy = y**2
    thres = np.mean(energy)
    new_signal = []
    i = 0
        
    if size > N:
        size = int(N/10)
    
    while i < (N - size):
        if energy[i] > thres:
            new_signal += y[i:i + size].tolist()
            i += size
        else:
            i += 1
    new_signal = np.array(new_signal)
    return new_signal

def my_filter5(y, size):
    N = len(y)
    energy = y**2
    thres = np.mean(energy)
    new_signal = []
    i = 0

    if size > N:
        size = int(N/10)

    while i < (N - size):
        if np.mean(energy[i:i+size]) > thres:
            new_signal += y[i:i + size].tolist()
            i += size
        else:
            i += size
    new_signal = np.array(new_signal)
    return new_signal
