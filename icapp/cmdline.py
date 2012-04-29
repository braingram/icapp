#!/usr/bin/env python

import logging
import optparse
import os
import sys

import numpy

import ica
import fio
import subsample


def parse(args):
    parser = optparse.OptionParser( \
            usage='usage: %prog [options] audiofiles...')
    options = [
            [['-c', '--count'], {
                'help': 'Component is noise if on > count channels',
                'default': 3,
                'type': 'int',
                }],
            [['-C', '--clean'], {
                'help': 'Also clean input files',
                'default': False,
                'action': 'store_true',
                }],
            [['-d', '--dtype'], {
                'help': 'data type to read from file',
                'default': 'float64',
                }],
            [['-i', '--icafile'], {
                'help': 'icafile to load/save',
                'default': 'ica.p',
                }],
            [['-m', '--method'], {
                'help': 'subsample method: simple, random',
                'default': 'simple',
                }],
            [['-n', '--ncomponents'], {
                'help': 'number of ica components',
                'type': 'int',
                }],
            [['-o', '--overwrite'], {
                'help': 'overwrite the icafile if it already exists',
                'default': False,
                'action': 'store_true',
                }],
            [['-O', '--output'], {
                'help': 'output directory',
                'default': 'cleaned',
                }],
            [['-p', '--plot'], {
                'help': 'generate debugging plots',
                'default': False,
                'action': 'store_true',
                }],
            [['-s', '--sargs'], {
                'help': 'subsample arguments',
                'action': 'append',
                'default': [],
                }],
            [['-S', '--chunksize'], {
                'help': 'Number of samples (per file) to reproject per chunk',
                'default': 44100,
                'type': 'int',
                }],
            [['-t', '--threshold'], {
                'help': 'Threshold to consider a mixing matrix cell active',
                'default': None,
                'type': 'float',
                }],
            [['-T', '--stdthreshold'], {
                'help': 'N stds used to auto-calculate threshold',
                'default': 1.0,
                'type': 'float',
                }],
            [['-v', '--verbose'], {
                'help': 'enable verbose output',
                'default': False,
                'action': 'store_true',
                }],
            ]

    [parser.add_option(*a, **kw) for (a, kw) in options]

    options, args = parser.parse_args(args)

    if options.ncomponents is None:
        options.ncomponents = len(args)

    return options, args


def run(args=None):
    if args is None:
        args = sys.argv[1:]

    options, files = parse(args)

    if len(files) < 2:
        raise ValueError("Must supply at least 2 files")

    if not all([os.path.exists(f) for f in files]):
        raise IOError("One (or more) input files does not exist: %s" % files)

    clean_files(files, **options_to_kwargs(options))


def options_to_kwargs(options):
    # -- all optional --
    # [ -c <count> -d <dtype> -m <method> -n <ncomponents> -o <overwrite>
    #    -p <plot> -s <sargs> -S <chunksize> -t <threshold> -v <verbose>]
    # modes:
    #  <files> -i <icafile> [] : just make ica file
    #  <files> -i <icafile> -CO <output> : use/make ica file and clean files
    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if (not (options.clean or options.overwrite)) and \
            os.path.exists(options.icafile):
        # this means "do nothing"
        logging.debug("Doing nothing")
        raise ValueError("These options do nothing. No cleaning, "
                "and no ica overwrite with existing ica file")

    kwargs = {}  # all should have default values

    kwargs['dtype'] = numpy.dtype(options.dtype)
    logging.debug("Setting dtype: %s" % kwargs['dtype'])

    for k in ['count', 'clean', 'icafile', 'method', 'ncomponents',
            'overwrite', 'output', 'plot', 'sargs', 'chunksize',
            'threshold', 'stdthreshold']:
        kwargs[k] = getattr(options, k)
        logging.debug("Settings %s: %s" % (k, kwargs[k]))

    # if cleaning files, make output directory
    if options.clean and (not os.path.exists(options.output)):
        os.makedirs(options.output)
    return kwargs


def process_src(src, method=None, sargs=None, ncomponents=None, count=None,
        threshold=None, stdthreshold=None):
    logging.debug("Subsample: %s, %s" % (method, sargs))
    data = subsample.subsample(src, method, *sargs)

    logging.debug("Making ica: %s, %s, %s" % \
            (ncomponents, count, threshold))
    #mm, um, cm, count, threshold = ica.make_ica(data, ncomponents, \
    #        count, threshold, stdthreshold)
    return ica.make_ica(data, ncomponents, count, threshold, stdthreshold)


def clean_files(filenames, **kwargs):
    logging.debug("Cleaning files: %s" % filenames)

    sfns = [os.path.basename(f) for f in filenames]
    logging.debug("Shortened filenames to: %s" % sfns)

    src, ext = fio.make_src(filenames)

    # open files
    icafile = kwargs['icafile']
    if not (os.path.exists(icafile)) or kwargs['overwrite']:
        logging.debug("Calculating ica...")

        # process source
        pkwargs = {}
        for k in ['method', 'sargs', 'ncomponents', 'count', 'threshold', \
                'stdthreshold']:
            pkwargs[k] = kwargs[k]
        mm, um, cm, count, threshold = process_src(src, **pkwargs)

        # save ica file
        logging.debug("Saving ica: %s" % icafile)
        fio.save_ica(icafile, mm, um, cm, filenames, count, threshold)
    else:
        logging.debug("Loading existing ica: %s" % icafile)
        mm, um, cm, fns, count, threshold = fio.load_ica(icafile)

        logging.debug("Checking ica")
        if (count != kwargs['count']):
            raise ValueError("icafile count[%s] != count[%s]" %
                    (count, kwargs['count']))
        if (fns != sfns):
            raise ValueError("icafile filenames do not match filenames: "
                "%s, %s" % (fns, sfns))
        if (kwargs['threshold'] is not None) and \
                (threshold != kwargs['threshold']):
            raise ValueError("icafile threshold[%s] != threshold[%s]" %
                    (threshold, kwargs['threshold']))

    if kwargs['clean']:
        logging.debug("Making data sink")
        sink = fio.make_sink(src, filenames, kwargs['output'])

        logging.debug("Cleaning data...")
        maxn = len(src)
        src.seek(0)
        sample = 0
        for n in fio.chunk(maxn, kwargs['chunksize']):
            logging.debug("Processing %i to %i" % (sample, sample + n))
            sample += n
            sink.write(ica.clean_data(src.read(n), cm))
        sink.close()

    logging.debug("Closing")
    src.close()
