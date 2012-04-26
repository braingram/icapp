#!/usr/bin/env python

import logging
import pickle
import warnings

import numpy


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import scikits.audiolab

datakeys = ['mix', 'unmix', 'clean', 'fns', 'count', 'thresh']


# --------------- loading and saving -----------------
def load_ica(filename):
    i = load_ica_dict(filename)
    return [i[k] for k in datakeys]


def load_ica_dict(filename):
    return pickle.load(open(filename, 'r'))


def save_ica(filename, mix, unmix, clean, filenames, count, threshold):
    save_ica_dict(filename, dict(zip(datakeys, \
            [mix, unmix, clean, filenames, count, threshold])))


def save_ica_dict(filename, ica_dict):
    with open(filename, 'w') as F:
        pickle.dump(ica_dict, F)


def read_files(filenames):
    pass


def chunk(n, chunksize, overlap=0):
    """
    Chunk generator
    """
    for i in xrange((n / chunksize) + 1):
        if (i * chunksize) >= n:
            return
        if ((i + 1) * chunksize + overlap) < n:
            yield (i * chunksize, (i + 1) * chunksize + overlap)
        else:
            yield (i * chunksize, n)


def new_chunk(n, chunksize):
    """
    Write a new chuncker here that just spits out how much to read
    so I can go:
    f.seek(0)
    for c in chunk(f.nframes, 1000):
        f.read(c)
    and not go beyond the edge of the file
    """
    i = 0
    while (i + chunksize < n):
        yield


class AudioFile(scikits.audiolab.Sndfile):
    read = scikits.audiolab.Sndfile.read_frames
    write = scikits.audiolab.Sndfile.write_frames


class MultiAudioFile(object):
    def __init__(self, filenames, **kwargs):
        self.filenames = filenames
        self.files = [AudioFile(f, **kwargs) for f in filenames]

    def seek(self, n):
        [f.seek(n) for f in self.files]

    def read(self, n):
        return numpy.vstack([f.read(n) for f in self.files])

    def write(self, data):
        [f.write(datum) for (datum, f) in zip(data, self.files)]


def clean_files(audioFiles, outputFiles, UM, remixer, chunksize):
    """
    Parameters
    ----------
    audioFiles : list of scikits.audiolab.Sndfile
        Should be open
    outputFiles : list of scikits.audiolab.Sndfile
        Should be open
    """

    CM = remixer * UM

    logging.debug("cleaning files, chunksize: %i" % chunksize)
    nframes = audioFiles[0].nframes
    for (s, e) in chunk(nframes, chunksize):
        logging.debug("chunk: %i to %i" % (s, e))
        data = []
        for infile in audioFiles:
            infile.seek(s)
            data.append(infile.read_frames(e - s))
        #tdata = ica.transform(np.array(data))

        #tdata = UM * np.array(data)
        #cdata = np.array(remixer * tdata)

        cdata = np.array(CM * np.array(data))
        for (cd, outfile) in zip(cdata, outputFiles):
            outfile.write_frames(cd)
            outfile.sync()
