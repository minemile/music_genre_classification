import librosa
import numpy as np
from server.database_wrapper import PostgresqlWrapper
from server.utils import Util
from sklearn.preprocessing import StandardScaler
import pandas as pd


class FeatureExtractor(object):
    def __init__(self, data):
        self.data = data
        self.std_scaler = StandardScaler()

    def generate_mfcc(self, n_mfcc, sr=22050):
        mfcc_means = np.empty((len(self.data), n_mfcc))

        # Generate mfcc means matrix MxN_MFCC
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got mfcc for {0} songs".format(i))
            mfcc = librosa.feature.mfcc(song, sr=sr, n_mfcc=n_mfcc)
            mfcc_scaled = self.std_scaler.fit_transform(mfcc)
            mfcc_mean = mfcc_scaled.mean(axis=1)
            mfcc_means[i] = mfcc_mean

        template = "mfcc_mean_{0}"
        col_names = []
        for i in range(n_mfcc):
            col_names.append(template.format(i))
        return mfcc_means

    def generate_zero_crossing_rate(self, frame_size, hop_length):
        zero_crossing_rates_std = np.empty((len(self.data), 1))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got zero_cross_rate for {0} songs".format(i))
            zcr = librosa.feature.zero_crossing_rate(song)
            zero_crossing_rates_std[i] = zcr.std()
        return zero_crossing_rates_std



if __name__ == '__main__':
    db = PostgresqlWrapper()
    util = Util()
    data_ = db.fetch_songs(150, 50, genres=['classical', 'metal', 'blues', 'country', 'jazz'])
    songs, genres = util.to_dataset(data_)
    extractor = FeatureExtractor(songs)
    df_mfcc = extractor.generate_mfcc(20)
    print(df_mfcc)
