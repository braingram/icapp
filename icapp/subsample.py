#!/usr/bin/env python

import numpy


def subsample(src, method, *args):
    """
    Return a subsample of data in src

    Parameters
    ----------
    src : file-like-object
        A data source that supports: seek, read, and __len__

    method : string
        The subsampling method to use

    args : tuple
        Arguments to the subsampling method


    Returns
    -------
    data : numpy.array
        Subsampled data


    Examples
    --------
    subsample(src, 'simple)
        get all samples in [0, len(src))

    subsample(src, 'simple', 10)
        get the first 10 samples

    subsample(src, 'simple', 10, 20)
        get samples [10, 20)

    subsample(src, 'random', 10)
        get 10 samples selected randomly within [0, len(src))

    subsample(src, 'random', 10, 20)
        get 10 samples selected randomly within [0, 20)

    subsample(src, 'random', 10, 20, 30)
        get 10 samples selected randomly within [20, 30)
    """
    if method == 'simple':
        if len(args) == 0:
            start = 0
            end = len(src)
        elif len(args) == 1:
            start = 0
            end = int(args[0])
        elif len(args) == 2:
            start = int(args[0])
            end = int(args[1])
        elif len(args) > 2:
            raise ValueError("Too many(>2) subsample arguments: %s" % \
                    (list(args)))
        src.seek(start)
        return src.read(end - start)
    elif method == 'random':
        if len(args) < 1:
            raise ValueError("Too few(<2) subsample arguments: %s" % \
                    (list(args)))
        elif len(args) == 1:
            n = int(args[0])
            minn = 0
            maxn = len(src)
        elif len(args) == 2:
            n = int(args[0])
            minn = 0
            maxn = int(args[1])
        elif len(args) == 3:
            n = int(args[0])
            minn = int(args[1])
            maxn = int(args[2])
        elif len(args) > 3:
            raise ValueError("Too many(>3) subsample arguments: %s" % \
                    (list(args)))

        def get(i):
            src.seek(i)
            return src.read(1)
        return numpy.hstack([get(i) for i in \
                numpy.random.randint(minn, maxn, n)])
    elif method == 'multirange':
        raise NotImplementedError
        #return numpy.vstack([numpy.arange(*r) for r in ranges])
    else:
        raise ValueError("Unknown method: %s" % method)
