
# coding: utf-8

# In[6]:

import librosa
import numpy as np


# In[16]:

def normalizedVar(Y):
    var_p = np.sqrt(Y.var(1))
    var_p = var_p.reshape(var_p.shape[0],1)   
    return(var_p)


# In[3]:

def varTrustFunc(Y, numFrames=200):
    
    x = int(Y.shape[1]/numFrames)
    varTrust = np.ones(Y.shape)
    
    for i in range(0, Y.shape[1], numFrames):
        indiceStart = i
        indiceStop = indiceStart+numFrames
        Y_p = Y[:,indiceStart:indiceStop]

        var_p = normalizedVar(Y_p)
        varTrust[:,indiceStart:indiceStop] = var_p
        
#     Y_p = Y[:,indiceStop:]
#     var_p = normalizedVar(Y_p)
#     varTrust[:,indiceStop:] = var_p
    varTrust = varTrust / np.max(varTrust)
    return(varTrust)


# In[4]:

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
    
#     x = 0.8
#     mask[mask < x] = x
    
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
#     Y_rec = np.multiply(newmag, np.exp(np.multiply(phase, (1j))))

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

#     mag, phase = librosa.magphase(Y)
#     newmag = librosa.db_to_amplitude(np.add(mask, Y_dB))
#     Y_rec = np.multiply(newmag, np.exp(np.multiply(phase, (1j))))

    
    y_rec = librosa.istft(Y_rec, hop_length=512)

    return y_rec
