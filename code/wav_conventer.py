import librosa


def sound_to_wav(path):
    y, sr = librosa.load(path)
    librosa.output.write_wav(path + '.wav', y, 22050)

path = '/Users/felipefelix/USP/tcc/code/Rexp/100041.wav'
sound_to_wav(path)
