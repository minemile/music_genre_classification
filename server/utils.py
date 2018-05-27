import numpy as np

class Util(object):

    def to_dataset(self, data):
        songs = []
        genres = []
        n_song = 1
        for batch in data:
            for _, genre, song in batch:
                if n_song % 50 == 0:
                    print('Prepared %s songs' % n_song)
                songs.append(song)
                genres.append(genre)
                n_song += 1
        return songs, genres

    def vector_to_features(self, vector):
        """ Extract median, mean, max, min, std from vector """
        return np.array([np.median(vector), vector.mean(), vector.std(), vector.min(), vector.max()])
