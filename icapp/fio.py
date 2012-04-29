#!/usr/bin/env python

import os
import pickle
import warnings

import numpy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import scikits.audiolab


datakeys = ['mm', 'um', 'cm', 'fns', 'c', 't']


# --------------- loading and saving -----------------
def load_ica(filename, key=None):
    """
    Parameters
    ----------
    filename : str
        Saved ica pickle file (e.g. ica.p)

    key : str (optional)
        ica data structure key to load


    Returns
    -------
    ica_struct : tuple
        Items in this tuple defined by datakeys
        If key is not None, returns just they keyed item


    Examples
    --------
    mm, um, cm, fns, c, t = load_ica('ica.p')
        Loads all items from ica.p including
            mm : mixing matrix
            um : unmixing matrix
            cm : cleaning matrix
            fns : shortened filenames
            c : cleaning count
            t : cleaning threshold

    cm = load_ica('ica.p', key='cm')
        Loads just the cleaning matrix
    """
    i = load_ica_dict(filename)
    if key is None:
        return [i[k] for k in datakeys]
    else:
        return i[k]


def load_ica_dict(filename):
    return pickle.load(open(filename, 'r'))


def save_ica(filename, mix, unmix, clean, filenames, count, threshold):
    # shorten filenames to basenames
    sfns = [os.path.basename(f) for f in filenames]
    save_ica_dict(filename, dict(zip(datakeys, \
            [mix, unmix, clean, sfns, count, threshold])))


def save_ica_dict(filename, ica_dict):
    with open(filename, 'w') as F:
        pickle.dump(ica_dict, F)


def chunk(n, chunksize):
    """
    Write a new chuncker here that just spits out how much to read
    so I can go:
    f.seek(0)
    for c in chunk(f.nframes, 1000):
        f.read(c)
    and not go beyond the edge of the file
    >>> list(new_chunk(10, 3))
    [3, 3, 3, 1]
    >>> sum(new_chunk(10, 3))
    10
    """
    i = 0
    while (i + chunksize < n):
        yield chunksize
        i += chunksize
    yield n - i


class AudioFile(scikits.audiolab.Sndfile):
    read = scikits.audiolab.Sndfile.read_frames
    write = scikits.audiolab.Sndfile.write_frames

    def __len__(self):
        return self.nframes


class MultiAudioFile(object):
    def __init__(self, filenames, **kwargs):
        assert len(filenames) > 0
        self.filenames = filenames
        self.files = [AudioFile(f, **kwargs) for f in filenames]
        if 'mode' in kwargs and kwargs['mode'] == 'w':
            return
        self._format = self.files[0].format
        self._samplerate = self.files[0].samplerate
        self._len = len(self.files[0])
        for f in self.files:
            assert len(f) == self._len
            assert f.format == self._format
            assert f.samplerate == self._samplerate

    def seek(self, n):
        [f.seek(n) for f in self.files]

    def read(self, n):
        return numpy.vstack([f.read(n) for f in self.files])

    def write(self, data):
        [f.write(datum) for (datum, f) in zip(data, self.files)]

    def __len__(self):
        return self._len

    def close(self):
        [f.close() for f in self.files]

    def get_format(self):
        return self._format

    format = property(get_format)

    def get_samplerate(self):
        return self._samplerate

    samplerate = property(get_samplerate)

    def sync(self):
        [f.sync() for f in self.files]


def make_src(filenames, **kwargs):
    exts = [os.path.splitext(f)[1].lower() for f in filenames]
    ext = exts[0]
    assert all([e == ext for e in exts]), "%s" % ([exts])

    if ext == '.wav':
        return MultiAudioFile(filenames), ext
    else:
        raise ValueError("Unknown extension: %s" % ext)


def make_sink(src, filenames, output_directory):
    ofiles = [os.path.join(output_directory, os.path.basename(f)) \
            for f in filenames]
    if isinstance(src, MultiAudioFile):
        dformat = src.format
        samplerate = src.samplerate
        return MultiAudioFile(ofiles, mode='w', format=dformat, channels=1, \
                samplerate=samplerate)
    else:
        raise ValueError("Unknown src class: %s" % src.__class__)
