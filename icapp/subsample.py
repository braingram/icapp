#!/usr/bin/env python

import numpy


# a simple subsample, also does range
simple = numpy.arange


def random(nsamps, maxn):
    """Take nsamps uniform random samples from [0,maxn)"""
    return numpy.random.randint(0, maxn, nsamps)


def multi_range(ranges):
    """Generate indices for many ranges"""
    return numpy.vstack([numpy.arange(*r) for r in ranges])
