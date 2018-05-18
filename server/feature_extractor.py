import librosa
import numpy as np
from .utils import Util
from sklearn.preprocessing import StandardScaler
import pandas as pd
from numpy.linalg import det
from multiprocessing.dummy import Pool as ThreadPool
#from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

class FeatureAggregator(object):

    def __init__(self, data, parallel=False):
        self.data = data
        self.parallel = parallel

    def get_features(self, feature_to_extract=None):
        generators = [MFCC, ZeroCrossing, Centroids, Rhythm, Energy]
        if self.parallel:
            with ThreadPoolExecutor(len(generators)) as pool:
                parallel_results = pool.map(self.generate, generators)
            result = np.hstack(parallel_results)
        else:
            result = [np.hstack([generator(self.data).generate()
                                 for generator in generators])]
        return result

    def generate(self, generator):
        result = generator(self.data).generate()
        print("Done ", generator)
        return result



class FeatureExtractor(object):

    def __init__(self, data, verbose=True, sr=22050):
        self.data = data
        self.util = Util()
        self.verbose = verbose
        self.sr = sr

    def generate(self):
        raise NotImplementedError()


class MFCC(FeatureExtractor):

    def __init__(self, data, sr=22050, n_mfcc=20):
        self.n_mfcc = n_mfcc
        super().__init__(data, sr)

    def generate(self):
        mfcc_means = np.empty((len(self.data), self.n_mfcc + 1))

        # Generate mfcc means matrix MxN_MFCC
        for i, song in enumerate(self.data):
            if self.verbose and i % 100 == 0:
                print("Got mfcc for {0} songs".format(i))
            mfcc = librosa.feature.mfcc(song, sr=self.sr, n_mfcc=self.n_mfcc)
            #mfcc_scaled = self.std_scaler.fit_transform(mfcc)
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_means[i] = np.append(mfcc_mean, det(
                np.cov(mfcc, rowvar=True)))
        return mfcc_means


class ZeroCrossing(FeatureExtractor):

    def generate(self):
        zero_crossing_rates = np.empty((len(self.data), 5))
        for i, song in enumerate(self.data):
            if self.verbose and i % 100 == 0:
                print("Got zero_cross_rate for {0} songs".format(i))
            zcr = librosa.feature.zero_crossing_rate(song)
            features = self.util.vector_to_features(zcr)
            zero_crossing_rates[i] = features
        return zero_crossing_rates


class Centroids(FeatureExtractor):

    def generate(self):
        """ Generate centroids, rolloff features """
        centroid_meanstd = np.empty((len(self.data), 5))
        rolloff_meanstd = np.empty((len(self.data), 5))
        for i, song in enumerate(self.data):
            if self.verbose and i % 100 == 0:
                print("Got centroid data for {0} songs".format(i))
            cent = librosa.feature.spectral_centroid(y=song, sr=self.sr)
            centroid_features = self.util.vector_to_features(cent)
            centroid_meanstd[i] = centroid_features
            rolloff = librosa.feature.spectral_rolloff(
                y=song, sr=self.sr, roll_percent=0.85)[0]
            rolloff_features = self.util.vector_to_features(rolloff)
            rolloff_meanstd[i, :] = rolloff_features

        result = np.hstack([centroid_meanstd, rolloff_meanstd])
        return result


class Rhythm(FeatureExtractor):

    def generate(self):
        rhythm_bpm = np.empty((len(self.data), 8))
        for i, song in enumerate(self.data):
            if self.verbose and i % 100 == 0:
                print("Got rhythm data for {0} songs".format(i))
            oenv = librosa.onset.onset_strength(y=song, sr=self.sr)
            # tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=self.sr)
            # dtempo = librosa.beat.tempo(
            #     onset_envelope=onset_env, sr=self.sr, aggregate=None)
            tempogram = librosa.feature.tempogram(
                onset_envelope=oenv, sr=self.sr)
            tempogram_features = self.util.vector_to_features(tempogram)
            ac_global = librosa.autocorrelate(
                oenv, max_size=tempogram.shape[0])
            ac_global = librosa.util.normalize(ac_global)
            tempo = librosa.beat.tempo(onset_envelope=oenv, sr=self.sr)
            rhythm_bpm[i] = np.hstack([tempo, ac_global.mean(), ac_global.std(), tempogram_features])
        return rhythm_bpm


class Energy(FeatureExtractor):

    def generate(self):
        flux_meanstd = np.empty((len(self.data), 5))
        low_energy = np.empty((len(self.data), 1))
        chroma_feature = np.empty((len(self.data), 12))
        for i, song in enumerate(self.data):
            if self.verbose and i % 100 == 0:
                print("Got rmse data for {0} songs".format(i))
            rmse = librosa.feature.rmse(y=song)
            mean_rmse = rmse.mean()
            low_energy[i] = np.sum(rmse <= mean_rmse) / len(rmse)

            max_rmse = rmse.max()
            rmse = np.where(rmse == 0, 1, rmse)
            D = librosa.stft(y=song) / rmse
            flux = np.sqrt((np.abs((D[:, 1:] - D[:, :-1]))**2).sum(axis=0))
            flux_features = self.util.vector_to_features(flux)
            flux_meanstd[i] = flux_features

            chroma = librosa.feature.chroma_cens(y=song, sr=self.sr)
            chroma_feature[i] = chroma.mean(axis=1)
        result = np.hstack([chroma_feature, low_energy, flux_meanstd])
        return result
