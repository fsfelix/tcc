import os
import time
import librosa
import matplotlib
import numpy as np
import pr_util as util

from scipy import signal, ndimage

def autodetec(y, power, thres, signal_or_change = 'both'):
    N = len(y)
    y_env = np.abs(y) ** power
    y_new = y.copy()
    print('máximo do envelope {}'.format(np.max(y_env)))
    print('mínimo do envelope {}'.format(np.min(y_env)))
    print('média do envelope {}'.format(np.mean(y_env)))
    thres = np.mean(y_env) * 10
    y2 = np.zeros(N)
    menor = 0
    maior = 0


    y2[y_env <= thres] = 1
    y2[y_env  > thres] = 2

    ## y3: { 2 = silence; 3 = change, 4 = signal}

    y3 = (y2[:-1] + y2[1:]).copy()

    y3[0] = 3
    y3[-1] = 3

    if signal_or_change == 'change':
        y_new[np.where(y3 != 3)[0]] = 0
        y_new[np.where(y3 == 3)[0]] = 1

        #return np.where(y3 == 3)[0]
        return y_new
    elif signal_or_change == 'signal':
        y_new[np.where(y3 != 4)[0]] = 0
        y_new[np.where(y3 == 4)[0]] = 1

        #return np.where(y3 == 4)[0]
        return y_new
    elif signal_or_change == 'both':
        y_new[np.where(y3 != 3)[0]] = 0
        y_new[np.where(y3 != 4)[0]] = 0
        y_new[np.where(y3 == 3)[0]] = 1
        y_new[np.where(y3 == 4)[0]] = 1

        return y_new


def estimate_struc_size(y, min_size = 0):
    print("estimate_struc_size")
    non_zeros = np.where(y == 1)[0]
    max_dis = -np.inf
    distances = []
    N = len(non_zeros)
    for i in range(N - 1):
        current_dis = non_zeros[i + 1] - non_zeros[i]
        if current_dis > min_size:
            distances += [current_dis]
    distances = np.array(distances)
    return int(np.percentile(distances, 99))

def dilate_both_and(y_new):
    struc_size = estimate_struc_size(y_new)
    print("struc size in dilation: {}".format(struc_size))
    struc_size *= 2
    struc_size -= 1
    half_size = int(struc_size/2)
    struc_l = np.array([1] * struc_size)
    struc_l[:half_size] = 0
    struc_r = np.array([1] * struc_size)
    struc_r[half_size:] = 0
    #plt.ylim([-2,2])
    y_filtered_l = ndimage.morphology.binary_dilation(y_new, struc_l, iterations=10)
    #plt.plot(y_filtered_l)
    y_filtered_r = ndimage.morphology.binary_dilation(y_new, struc_r, iterations=10)
    #plt.plot(y_filtered_r)
    y_filtered = np.logical_and(y_filtered_l, y_filtered_r)
    #plt.plot(y_filtered)
    return y_filtered

def durations_syllable(y):
    durations = []
    N = len(y)
    i = 0
    init = 0
    end = 0
    syllable = y[0] == 1

    while (i < N):
        if (y[i] == 1 and not syllable):
            init = i
            syllable = True
        if (y[i] == 0 and syllable):
            end = i - 1
            durations += [end - init]
            syllable = False
        i += 1
    return durations

def filter_durations(durations, dur_min, dur_max):
    return [dur for dur in durations if dur < dur_max and dur > dur_min]

def get_syllable_durations(y, sr, min_dur, max_dur):
    y_new = autodetec(y, 2, 0.2, 'both')
    y_filtered = dilate_both_and(y_new)
    durations = durations_syllable(y_filtered)
    durations = filter_durations(durations, util.time_to_samples(min_dur, sr), util.time_to_samples(max_dur, sr))

    if durations != []:
        #durations = [util.samples_to_time(dur, sr) for dur in durations]
        #durations = np.array(durations)
        return util.samples_to_time(np.porcentile(durations, 90), sr)
    else:
        print("Duração não encontrada")
        return -1
    #return durations