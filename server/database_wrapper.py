import io
import os
import logging
import psycopg2
import numpy as np
import librosa

class PostgresqlWrapper(object):
    """ Postgresql wrapper to heroku server to upload and download music data"""
    DATABASE_URL = "postgres://phhcxzngavxxfh:85553a16e2880ac28faf01c256fcf50caa8a9b7326895562afe54a2b1f7e639d@ec2-54-75-239-237.eu-west-1.compute.amazonaws.com:5432/d5jk6qjst0rku1"
    MUSIC_PATH = "genres"

    def __init__(self):
        self.__init_logger()
        self.log.info("Connecting to database")
        self.conn = psycopg2.connect(self.DATABASE_URL, sslmode='require')
        self.cur = self.conn.cursor()
        self.register_adapters()
        self.create_table(False)

    def __init_logger(self):
        ch = logging.StreamHandler() # console
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        self.log.addHandler(ch)

    def create_table(self, clean=False):
        self.log.info("Creating table")
        if clean:
            self.cur.execute('drop table music')
            self.conn.commit()
        statement = "CREATE TABLE if not exists music \
        (id serial PRIMARY KEY, genre varchar(100), data BYTEA);"
        self.cur.execute(statement)

    def insert_song(self, genre, song):
        statement = "Insert into music(genre, data) values(%s, %s)"
        self.cur.execute(statement, (genre, song))

    def select_songs(self, genre=None):
        if genre is None:
            statement = "Select * from music"
        else:
            statement = "Select * from music where genre = %s"
        self.cur.execute(statement, (genre,))
        return self.cur.fetchall()

    def register_adapters(self):
        """ Handy adapters to transalte np.array to binary and vice versa """
        def _adapt_array(text):
            out = io.BytesIO()
            np.save(out, text)
            out.seek(0)
            return psycopg2.Binary(out.read())

        def _typecast_array(value, cur):
            if value is None:
                return None

            data = psycopg2.BINARY(value, cur)
            bdata = io.BytesIO(data)
            bdata.seek(0)
            return np.load(bdata)

        psycopg2.extensions.register_adapter(np.ndarray, _adapt_array)
        t_array = psycopg2.extensions.new_type(
            psycopg2.BINARY.values, "numpy", _typecast_array)
        psycopg2.extensions.register_type(t_array)
        self.log.info("Done register types")

    def to_database(self, folder=None, limit=100):
        """ Process music to database """
        for root, _, files in os.walk(self.MUSIC_PATH):
            genre = root.split('/')[-1]
            if folder is not None and folder != genre:
                continue
            count_music = 0
            for file_ in files:
                if count_music == limit:
                    break
                self.log.info("Inserting song %s", file_)
                song = librosa.load(os.path.join(root, file_))[0]
                self.insert_song(genre, song)
                count_music += 1
        self.conn.commit()

if __name__ == '__main__':
    db = PostgresqlWrapper()
    #db.to_database(limit=10)
    result = db.select_songs('jazz')
    print(result)
