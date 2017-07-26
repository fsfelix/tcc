import os
import numpy as np
import librosa
import utility_functions as util

def extract_local_features(genre, filename, beat_frames, feature_name, feature_function, y, **kwargs):
    feature = None

    feature_txt = os.path.join(util.LOCAL_FEATURE_DIR, genre, filename + "." + feature_name)
    if not os.path.isfile(feature_txt):
        print "creating " + filename + "." + feature_name
        feature = feature_function(y=y, **kwargs)
        np.savetxt(feature_txt, feature)

    feature_beat_txt = os.path.join(util.LOCAL_FEATURE_DIR, genre, filename + "." + feature_name + "_beat")
    if not os.path.isfile(feature_beat_txt):
        if feature is None:
            feature = feature_function(y=y, **kwargs)
        print "creating " + filename + "." + feature_name + "_beat"
        feature_beat = librosa.util.utils.sync(data=feature, idx=beat_frames)
        np.savetxt(feature_beat_txt, feature_beat)

def main():
    for genre in util.genres:
        for n in range(util.n_songs):
            filename = genre + "." + str(n).zfill(5)
            y, sr = librosa.load(os.path.join(util.AUDIO_DIR, genre, filename + ".au"), sr=None)

            y_harmonic, y_perc = librosa.effects.hpss(y)
            onset_envelope = librosa.onset.onset_strength(y=y_perc, sr=sr, aggregate=np.median)
            tempo, beat_frames = librosa.beat.beat_track(y=y_perc, sr=sr, onset_envelope=onset_envelope)

            #1
            extract_local_features(genre, filename, beat_frames, "chroma_stft", librosa.feature.chroma_stft, y, sr=sr)

            #2
            extract_local_features(genre, filename, beat_frames, "chroma_cqt", librosa.feature.chroma_cqt, y, sr=sr)

            #3
            extract_local_features(genre, filename, beat_frames, "chroma_cens", librosa.feature.chroma_cens, y, sr=sr)

            #4
            extract_local_features(genre, filename, beat_frames, "melspectrogram", librosa.feature.melspectrogram, y, sr=sr)

            #5
            extract_local_features(genre, filename, beat_frames, "mfcc", librosa.feature.mfcc, y, sr=sr)

            #6
            extract_local_features(genre, filename, beat_frames, "rmse", librosa.feature.rmse, y)

            #7
            extract_local_features(genre, filename, beat_frames, "spectral_centroid", librosa.feature.spectral_centroid, y, sr=sr)

            #8
            extract_local_features(genre, filename, beat_frames, "spectral_bandwidth", librosa.feature.spectral_bandwidth, y, sr=sr)

            #9
            extract_local_features(genre, filename, beat_frames, "spectral_contrast", librosa.feature.spectral_contrast, y, sr=sr)

            #10
            extract_local_features(genre, filename, beat_frames, "spectral_rolloff", librosa.feature.spectral_rolloff, y, sr=sr)

            #11
            extract_local_features(genre, filename, beat_frames, "poly_features", librosa.feature.poly_features, y, sr=sr)

            #12
            extract_local_features(genre, filename, beat_frames, "tonnetz", librosa.feature.tonnetz, y, sr=sr)

            #13
            extract_local_features(genre, filename, beat_frames, "zero_crossing_rate", librosa.feature.zero_crossing_rate, y)

            #14
            extract_local_features(genre, filename, beat_frames, "tempogram", librosa.feature.tempogram, y, sr=sr)

if __name__ == "__main__":
    main()
