import os
import librosa
import itertools

import pandas as pd
import numpy as np
from joblib import load


def get_features(y, sr, n_fft=1024, hop_length=512):
    # Features to concatenate in the final dictionary
    features = {'centroid': None, 'roloff': None, 'flux': None, 'rmse': None,
                'zcr': None, 'contrast': None, 'bandwidth': None, 'flatness': None}

    # Count silence
    if 0 < len(y):
        y_sound, _ = librosa.effects.trim(
            y, frame_length=n_fft, hop_length=hop_length)
    features['sample_silence'] = len(y) - len(y_sound)

    # Using librosa to calculate the features
    features['centroid'] = librosa.feature.spectral_centroid(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['roloff'] = librosa.feature.spectral_rolloff(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['zcr'] = librosa.feature.zero_crossing_rate(
        y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['rmse'] = librosa.feature.rms(
        y, frame_length=n_fft, hop_length=hop_length).ravel()
    features['flux'] = librosa.onset.onset_strength(y=y, sr=sr).ravel()
    features['contrast'] = librosa.feature.spectral_contrast(y, sr=sr).ravel()
    features['bandwidth'] = librosa.feature.spectral_bandwidth(
        y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel()
    features['flatness'] = librosa.feature.spectral_flatness(
        y, n_fft=n_fft, hop_length=hop_length).ravel()

    # MFCC treatment
    mfcc = librosa.feature.mfcc(
        y, n_fft=n_fft, hop_length=hop_length, n_mfcc=19)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()

    # Get statistics from the vectors
    def get_moments(descriptors):
        result = {}
        for k, v in descriptors.items():
            result['{}_mean'.format(k)] = np.mean(v)
            result['{}_var'.format(k)] = np.var(v)
        return result

    dict_agg_features = get_moments(features)
    dict_agg_features['tempo'] = librosa.beat.tempo(y, sr=sr)[0]
    dict_agg_features = pd.Series(dict_agg_features)
    dict_agg_features = pd.Series.to_frame(dict_agg_features).transpose()

    return dict_agg_features
