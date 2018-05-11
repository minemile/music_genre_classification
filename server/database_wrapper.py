import io
import os
import logging
from multiprocessing.dummy import Pool
from pprint import pprint

import psycopg2
from psycopg2.pool import SimpleConnectionPool
import numpy as np
import librosa


class PostgresqlWrapper(object):
    """ Postgresql wrapper to heroku server to upload and download music data"""
    DATABASE_URL = "postgres://phhcxzngavxxfh:85553a16e2880ac28faf01c256fcf50caa8a9b7326895562afe54a2b1f7e639d@ec2-54-75-239-237.eu-west-1.compute.amazonaws.com:5432/d5jk6qjst0rku1"
    MUSIC_PATH = "genres"
    LOCALHOST_STING = "host='localhost' dbname='music' user='meudon' password='123'"

    def __init__(self, conn_num=3):
        self.__init_logger()
        self.log.info("Creating pool")
        self.conn_num = conn_num
        self.conn = psycopg2.connect(self.LOCALHOST_STING)
        self.cur = self.conn.cursor()
        self.pool = SimpleConnectionPool(
            self.conn_num, self.conn_num + 5, self.LOCALHOST_STING)
        self.register_adapters()
        self.create_table(False)

    def __init_logger(self):
        ch = logging.StreamHandler()  # console
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.log = logging.getLogger(__name__)
        #if (self.log.hasHandlers()):
        #    self.log.handlers.clear()
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
        self.conn.commit()

    def insert_song(self, genre, song):
        statement = "Insert into music(genre, data) values(%s, %s)"
        self.cur.execute(statement, (genre, song))

    def select_songs(self, limit=None, offset=None, genre=None):
        conn = self.pool.getconn()
        cur = conn.cursor()
        if genre is None:
            statement = "Select * from music order by id limit %s offset %s"
            self.log.info("Statement %s", statement % (limit, offset))
            cur.execute(statement, (limit, offset))
            self.log.info("Done with %s", statement % (limit, offset))
        else:
            statement = "Select * from music where genre = %s order by id limit %s offset %s"
            self.log.info("Statement %s", statement % (genre, limit, offset))
            cur.execute(statement, (genre, limit, offset))
        db_result = cur.fetchall()
        cur.close()
        self.pool.putconn(conn)
        return db_result

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

    def to_database(self, folders=None, limit=1000):
        """ Process music to database """
        for root, _, files in os.walk(self.MUSIC_PATH):
            genre = root.split('/')[-1]
            if folders is not None and genre not in folders:
                continue
            for i, file_ in enumerate(files):
                if i == limit:
                    break
                self.log.info("Inserting song %s", file_)
                song = librosa.load(os.path.join(root, file_))[0]
                self.insert_song(genre, song)
        self.conn.commit()
        self.close_connection()

    def close_connection(self):
        self.log.info("Closing connection")
        self.pool.closeall()
        self.cur.close()
        self.conn.close()

    def fetch_songs(self, count, limit=50, genres=None):
        """ Fetch song in concurrent from database
            limit - how many song to fetch from one thread
            count - how many song to fetch
        """
        self.log.info("Start fetching %s songs", count)
        producer = []
        iter_ = 0
        offset = 0
        while offset < count:
            offset = limit * iter_
            if genres is not None:
                for genre in genres:
                    producer.append((limit, offset, genre))
            else:
                producer.append((limit, offset))
            iter_ += 1
        with Pool(self.conn_num) as pool:
            result = pool.starmap(self.select_songs, producer)
        return result


if __name__ == '__main__':
    db = PostgresqlWrapper(5)
    # Fetch 100 songs. acctually 150 ;)
    # Result format (ID, genre, np array song)
    songs = db.fetch_songs(10, genres=['classical', 'jazz', 'metal', 'pop'])
    pprint(songs)
