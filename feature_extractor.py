import librosa
import numpy as np
from server.database_wrapper import PostgresqlWrapper
from server.utils import Util
from sklearn.preprocessing import StandardScaler
import pandas as pd
from numpy.linalg import det


class FeatureExtractor(object):
    def __init__(self, data):
        self.data = data
        self.std_scaler = StandardScaler()

    def generate_mfcc(self, n_mfcc, sr=22050):
        mfcc_means = np.empty((len(self.data), n_mfcc + 1))

        # Generate mfcc means matrix MxN_MFCC
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got mfcc for {0} songs".format(i))
            mfcc = librosa.feature.mfcc(song, sr=sr, n_mfcc=n_mfcc)
            mfcc_scaled = self.std_scaler.fit_transform(mfcc)
            mfcc_mean = mfcc_scaled.mean(axis=1)
            mfcc_means[i] = np.append(mfcc_mean, det(np.cov(mfcc_scaled, rowvar = True)))

        template = "mfcc_mean_{0}"
        col_names = []
        for i in range(n_mfcc):
            col_names.append(template.format(i))
        return mfcc_means

    def generate_zero_crossing_rate(self, frame_size, hop_length):
        zero_crossing_rates= np.empty((len(self.data), 2))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got zero_cross_rate for {0} songs".format(i))
            zcr = librosa.feature.zero_crossing_rate(song)
            zero_crossing_rates[i] = np.array([zcr.mean(),zcr.std()])
        return zero_crossing_rates

    def generate_centoid_meanstd(self, sr=22050):
        '''
            Returns ndarray with 2 columns: 
                First column:  mean
                Second column: std
        '''
        centroid_meanstd = np.empty((len(self.data), 2))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got centroid data for {0} songs".format(i))
            cent = librosa.feature.spectral_centroid(y=song, sr=sr)
            centroid_meanstd[i, :] = np.array([cent.mean(), cent.std()]) 

        return centroid_meanstd
    
    def generate_rhythm(self, sr=22050):
        rhythm_bpm = np.empty((len(self.data), 4))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got rhythm data for {0} songs".format(i))
            onset_env = librosa.onset.onset_strength(y=song, sr=sr) 
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
            dtempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, aggregate=None)
            rhythm_bpm[i, :] = np.array([tempo, dtempo.mean(), dtempo.std(), len(np.unique(dtempo))]) 
        return rhythm_bpm
    
    def generate_rolloff(self, sr=22050):
        '''
            Returns ndarray with 2 columns: 
                First column:  mean
                Second column: std
        '''
        rolloff_meanstd = np.empty((len(self.data), 2))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got rolloff data for {0} songs".format(i))
            rolloff = librosa.feature.spectral_rolloff(y=song, sr=sr, roll_percent=0.85)[0]
            rolloff_meanstd[i, :] = np.array([rolloff.mean(), rolloff.std()]) 

        return rolloff_meanstd

    def generate_flux(self, sr=22050):
        '''
            Returns ndarray with 2 columns: 
                First column:  mean
                Second column: std
        '''
        flux_meanstd = np.empty((len(self.data), 2))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got flux data for {0} songs".format(i))
            rmse = librosa.feature.rmse(y=song)
            max_rmse = max(rmse)
            rmse = np.where(rmse == 0, 1, rmse)
            D = librosa.stft(y=song) / rmse
            flux = np.sqrt((np.abs((D[:, 1:] - D[:, :-1]))**2).sum(axis=0))
            flux_meanstd[i, :] = np.array([flux.mean(), flux.std()])

        return flux_meanstd

    def generate_energy(self, sr=22050):
        low_energy = np.empty(len(self.data))
        for i, song in enumerate(self.data):
            if i % 50 == 0: print("Got rmse data for {0} songs".format(i))
            rmse = librosa.feature.rmse(y=song)
            mean_rmse = rmse.mean()
            low_energy[i] = np.sum(rmse <= mean_rmse) / len(rmse)
        
        return low_energy.reshape(-1,1)

if __name__ == '__main__':
    db = PostgresqlWrapper()
    util = Util()
    data_ = db.fetch_songs(150, 50, genres=['classical', 'metal', 'blues', 'country', 'jazz'])
    songs, genres = util.to_dataset(data_)
    extractor = FeatureExtractor(songs)
    df_mfcc = extractor.generate_mfcc(20)
    print(df_mfcc)
