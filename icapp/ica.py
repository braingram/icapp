#!/usr/bin/env python

import logging

import numpy
from scikits.learn.decomposition import FastICA


def run_ica(data, ncomponents):
    """
    Parameters
    ----------
    data : numpy.array
        data to fit ica

    ncomponents : int
        number of components for ica


    Returns
    -------
    mix : numpy.matrix
        mixing matrix

    unmix : numpy.matrix
        un-mixing matrix


    See Also
    --------
    scikits.learn.decomposition.FastICA
    """
    logging.debug("running ica: %i" % ncomponents)
    ica = FastICA(ncomponents)
    ica.fit(data)
    if hasattr(ica, 'mixing_matrix'):
        return numpy.matrix(ica.mixing_matrix()), \
                numpy.matrix(ica.unmixing_matrix_)
    else:
        return numpy.matrix(ica.get_mixing_matrix()), \
                numpy.matrix(ica.unmixing_matrix_)


def clean_ica(mix, count, threshold=None, stdthreshold=1.0, full=False):
    """
    Parameters
    ----------
    mix : numpy.matrix
        ica mixing matrix

    count : int
        If ica components is super-threshold on > count channels remove it

    threshold : float (optional)
        Threshold used to binarize the mixing matrix, If None, autocompute

    full : bool (optional)
        Return full output


    Returns
    -------
    clean : numpy.matrix
        cleaned ica mixing matrix

    count : int (if full)
        same as parameter

    threshold : float (if full)
        same as parameter, may be autocalculated if parameter was None
    """
    if threshold is None:
        threshold = numpy.mean(mix) + numpy.std(mix) * stdthreshold
    logging.debug("cleaning ica: %f, %i" % (threshold, count))
    votes = numpy.sum(numpy.abs(numpy.array(mix)) > threshold, 0)
    logging.debug("component votes: %s" % votes)
    bad = numpy.where(votes > count)[0]
    logging.debug("noise components: %s" % bad)
    clean = mix.copy()
    clean[:, bad] = numpy.zeros_like(mix[:, bad])
    if full:
        return clean, count, threshold
    return clean


def make_cleaning_matrix(mix, unmix):
    """
    Parameters
    ----------
    mix : numpy.matrix
        ica mixing matrix

    unmix : numpy.matrix
        ica un-mixing matrix

    Returns
    -------
    clean : numpy.matrix
        Single matrix (combining unmix and mix) used to clean data
    """
    return numpy.matrix(mix) * numpy.matrix(unmix)


def clean_data(data, clean):
    """
    Parameters
    ----------
    data : numpy.array
        data to clean

    clean : numpy.matrix
        ica cleaning matrix (mix * unmix). See make_cleaning_matrix


    Returns
    -------
    cleaned : numpy.array
        cleaned data
    """
    return numpy.array(clean * data)


def make_ica(data, ncomponents, count, threshold, stdthreshold):
    """
    Combined running and cleaning


    Parameters
    ----------
    data : numpy.array
        data to fit ica

    ncomponents : int
        number of components for ica

    count : int
        If ica components is super-threshold on > count channels remove it

    threshold : float
        Threshold used to binarize the mixing matrix, If None, autocompute


    Returns
    -------
    mm : numpy.matrix
        ica mixing matrix

    mm : numpy.matrix
        ica un-mixing matrix

    cm : numpy.matrix
        ica cleaning matrix

    count : int
        If ica components is super-threshold on > count channels remove it

    threshold : float
        Threshold used to binarize the mixing matrix


    See Also
    --------
    run_ica
    clean_ica
    make_cleaning_ica
    """
    mm, um = run_ica(data, ncomponents)
    cmm, count, threshold = clean_ica(mm, count, threshold, stdthreshold, \
            full=True)
    cm = make_cleaning_matrix(cmm, um)
    return cmm, um, cm, count, threshold
