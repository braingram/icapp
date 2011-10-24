#!/usr/bin/env python

# tests
# 1) fake data
# - generate fake data
# - run ica
# - split and plot
#
# 2) real data
# - read in real data
# - run ica
# - split and plot
#
# basic parts
# 1) read audio files (type, names) and subsample
# 2) run ica
# 3) determine what parts are good/bad
# 4) remove bad parts/keep good parts
# 5) recombine and save

import glob, logging, optparse, os, sys

import numpy as np
import pylab as pl

import scikits.audiolab as al

from scikits.learn.decomposition import FastICA

def parse_options(args=None):
    """
    chain these like so:
        main parser -> get subsample method -> parse subsample options -> check options -> GO
    """
    parser = optparse.OptionParser(usage="usage: %prog [options] audiofiles...")
    parser.add_option("-m", "--method", dest = "method",
            help = "subsample method: simple,",
            default = "random", type = "string")

    parser.add_option("-M", "--mixingmatrix", dest = "mixingmatrix",
            help = "load a pre-computed mixingmatrix (must also define U)",
            default = "", type = "string")
    
    parser.add_option("-U", "--unmixingmatrix", dest = "unmixingmatrix",
            help = "load a pre-computed unmixingmatrix (must also define M)",
            default = "", type = "string")

    parser.add_option("-s", "--subsample", dest = "subsample",
            help = "subsample argument", action = "append",
            default = [])

    parser.add_option("-n", "--ncomponents", dest = "ncomponents",
            help = "number of ica components",
            default = 32, type = "int")
    
    parser.add_option("-d", "--dtype", dest = "dtype",
            help = "data type to read from file",
            default = "float64", type = "string")

    parser.add_option("-v", "--verbose", dest = "verbose",
            help = "enable verbose output",
            default = False, action = "store_true")
    
    parser.add_option("-p", "--plot", dest = "plot",
            help = "generate debugging plots",
            default = False, action = "store_true")

    parser.add_option("-o", "--output", dest = "output",
            help = "output directory",
            default = "cleaned")

    parser.add_option("-t", "--threshold", dest = "threshold",
            help = "Threshold at which to consider a mixing matrix cell active",
            default = None, type = "float")

    parser.add_option("-c", "--count", dest = "count",
            help = "Number of cells within a column of the mixing matrix required to count the component as noise",
            default = 3, type = "int")
    
    parser.add_option("-C", "--chunksize", dest = "chunksize",
            help = "Number of samples (per file) to reproject per chunk",
            default = 44100, type = "int")

    (options, args) = parser.parse_args(args)

    # check options
    if len(args) < 2:
        raise ValueError("Must supply at least two audio files[%s]" % str(args))

    options.dtype = np.dtype(options.dtype)
    if options.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    if not os.path.exists(options.output):
        os.makedirs(options.output)

    return options, args


def subsample(audioFiles, n):
    """
    This is a simple 'subsample' function which just reads the first n frames

    Parameters
    ----------
    audioFiles : list of scikits.audiolab.Sndfile
        Should all be open
    n : int
        Number of samples to read

    Returns
    -------
    data : numpy.ndarray
        Subsampled data (ordered: FILE x SAMPLE)
    
    Notes
    -----
    More complex functions could take it's place. Such as...
        1) uniform sampling throughout file
        2) random sampling throughout file
        3) oversampling of licking periods (or other artifact-known periods)
    """
    logging.debug("simple subsampling: %i" % n)
    [af.seek(0) for af in audioFiles]
    data = [af.read_frames(n) for af in audioFiles]
    return np.array(data)

def random_subsample(audioFiles, n):
    logging.debug("random subsampling: %i" % n)
    samps = np.random.randint(0, audioFiles[0].nframes, n)
    data = np.empty((len(audioFiles),n))
    for (si, s) in enumerate(samps):
        for (ai, af) in enumerate(audioFiles):
            af.seek(s)
            data[ai,si] = af.read_frames(1)
    return data

def range_subsample(audioFiles, start, end):
    logging.debug("range subsampling: %i %i" % (start, end))
    [af.seek(0) for af in audioFiles]
    data = [af.read_frames(end-start) for af in audioFiles]
    return np.array(data)

def multi_range_subsample(audioFiles, ranges):
    """
    ranges is a list of (start, stop) tuples
    """
    total = sum([r[1] - r[0] for r in ranges])
    data = np.empty((len(audioFiles), total))
    cursors = np.zeros(len(audioFiles))
    for (ri, r) in enumerate(ranges):
        for (ai, af) in enumerate(audioFiles):
            af.seek(r[0])
            data[ai,cursors[ai]+r[1]-r[0]] = af.read_frames(r[1]-r[0])
            cursors[ai] += r[1] - r[0]
    return data

def run_ica(data, ncomponents):
    logging.debug("running ica: %i" % ncomponents)
    ica = FastICA(ncomponents)
    ica.fit(data)
    return ica

def clean_ica(ica, threshold, count):
    mm = ica.get_mixing_matrix()
    if threshold is None:
        mf = mm.reshape(mm.shape[0] * mm.shape[1])
        threshold = np.mean(mf) + np.std(mf) * 2.
    logging.debug("cleaning ica: %f, %i" % (threshold, count))
    logging.debug("column maxes: %s" % str(np.max(np.abs(mm), 0)))
    votes = np.sum(np.abs(mm) > threshold, 0)
    logging.debug("component votes: %s" % str(votes))
    bad = np.where(votes > count)[0]
    logging.debug("noise components: %s" % str(bad))
    mm[:,bad] = np.zeros_like(mm[:,bad])
    return np.matrix(mm)
    try:
        umm = np.linalg.inv(mm)
    except np.linalg.LinAlgError:
        logging.debug("Failed to mathematically invert matrix, trying pseudo-inverse")
        umm = np.linalg.pinv(mm)
    return np.matrix(umm)

def make_output_files(inputFilenames, outputdir, auformat, samplerate):
    logging.debug("Making output files in directory %s" % outputdir)
    outputFiles = []
    for infile in inputFilenames:
        ofn = "%s/%s" % (outputdir, os.path.basename(infile))
        outputFiles.append(al.Sndfile(ofn, 'w', auformat, 1, samplerate))
    return outputFiles

def chunk(n, chunksize, overlap=0):
    """
    Chunk generator
    """
    for i in xrange((n/chunksize)+1):
        if (i * chunksize) >= n:
            return
        if ((i+1) * chunksize + overlap) < n:
            yield (i*chunksize, (i+1)*chunksize + overlap)
        else:
            yield (i*chunksize, n)

def clean_files(audioFiles, outputFiles, UM, remixer, chunksize):
    """
    Parameters
    ----------
    audioFiles : list of scikits.audiolab.Sndfile
        Should be open
    outputFiles : list of scikits.audiolab.Sndfile
        Should be open
    """
    logging.debug("cleaning files, chunksize: %i" % chunksize)
    nframes = audioFiles[0].nframes
    for (s, e) in chunk(nframes, chunksize):
        logging.debug("chunk: %i to %i" % (s, e))
        data = []
        for infile in audioFiles:
            infile.seek(s)
            data.append(infile.read_frames(e-s))
        #tdata = ica.transform(np.array(data))
        tdata = UM * np.array(data)
        cdata = np.array(remixer * tdata)
        for (cd, outfile) in zip(cdata, outputFiles):
            outfile.write_frames(cd)
            outfile.sync()

def process():
    options, inFilenames = parse_options()
    # open files
    afs = [al.Sndfile(f) for f in inFilenames]
    if (options.mixingmatrix.strip() == "") and (options.unmixingmatrix.strip() == ""):

        # subsample
        if options.method == 'simple':
            if len(options.subsample) == 0:
                args = [44100,] # set defaults
            elif len(options.subsample) == 1:
                try:
                    args = [int(options.subsample[0]),]
                except ValueError:
                    raise ValueError("Could not convert subsample argument to int[%s]" % options.subsample[0])
            else:
                raise ValueError("Wrong number of subsample arguments, expected 1: %s" % str(options.subsample))
            data = subsample(afs, *tuple(args))
        elif options.method == 'random':
            if len(options.subsample) == 0:
                args = [44100,]
            elif len(options.subsample) == 1:
                try:
                    args = [int(options.subsample[0]),]
                except ValueError:
                    raise ValueError("Could not convert subsample argument to int[%s]" % options.subsample[0])
            else:
                raise ValueError("Wrong number of subsample arguments, expected 1: %s" % str(options.subsample))
            data = random_subsample(afs, *tuple(args))
        elif options.method == 'range':
            if len(options.subsample) == 2: # only accept two arguments, a start and stop
                try:
                    args = [int(options.subsample[0]),int(options.subsample[1])]
                except ValueError:
                    raise ValueError("Could not convert subsample arguments to int[%s,%s]" % tuple(options.subsample))
            else:
                raise ValueError("Wrong number of subsample arguments, expected 2: %s" % str(options.subsample))
            data = range_subsample(afs, *tuple(args))
        elif options.method == 'multi':
            if (len(options.subsample) % 2) or (len(options.subsample) == 0):
                raise ValueError("Wrong number of subsample arguments, expected at least or multiples of 2: %s" % str(options.subsample))
            else:
                ranges = []
                for (s, e) in zip(options.subsample[::2], options.subsample[1::2]):
                    ranges.append((s,e))
            data = multi_range_subsample(afs, ranges)
        else:
            raise ValueError("Unknown subsample method: %s" % options.method)

        # ica
        ica = run_ica(data, options.ncomponents)

        # clean
        MM = clean_ica(ica, options.threshold, options.count)
        UM = pl.matrix(ica.unmixing_matrix_)
        # save M
        pl.savetxt("%s/mixingmatrix" % options.output, MM)
        pl.savetxt("%s/unmixingmatrix" % options.output, UM)
    else:
        logging.debug("Loading matrix from file: %s" % options.mixingmatrix)
        MM = pl.matrix(pl.loadtxt(options.mixingmatrix))
        logging.debug("Loading matrix from file: %s" % options.unmixingmatrix)
        UM = pl.matrix(pl.loadtxt(options.unmixingmatrix))

    ofs = make_output_files(inFilenames, options.output, afs[0].format, afs[0].samplerate)
    #clean_files(afs, ofs, ica, MM, options.chunksize)
    clean_files(afs, ofs, UM, MM, options.chunksize)

    # close
    logging.debug("Closing files")
    [ofile.close() for ofile in ofs]
    if options.plot:
        pl.figure()
        #pl.imshow(ica.get_mixing_matrix(), interpolation='none')
        pl.imshow(MM, interpolation='none')
        pl.colorbar()
        pl.suptitle("Mixing matrix (pre-cleaning)")
        pl.figure()
        pl.imshow(MM, interpolation='none')
        pl.colorbar()
        pl.suptitle("Mixing matrix (post-cleaning)")
        # plot components?
        pl.show()
    return UM, MM

if __name__ == '__main__':
    if len(sys.argv[1:]) == 0:
        sys.argv += glob.glob('data/clip_*')
        sys.argv.append('-v')
        sys.argv.append('-t')
        sys.argv.append('0.1')
    ica, M = process()
    sys.exit(0)
    
    # -------------------------------------
    # -------------  old code  ------------
    # -------------------------------------
    
    afs = sys.argv[1:]
    if len(afs) < 2:
        afs = glob.glob('data/clip_*')
        #raise ValueError("Must supply at least 2 audio files")

    # open audio files
    print "opening files"
    dfs = [al.Sndfile(f) for f in afs]
    #n = dfs[0].nframes
    n = 4410

    # read data
    print "reading data"
    data = subsample(dfs, n)
    #data = [df.read_frames(n) for df in dfs]
    #data = np.array(data) # ordered as FILE x SAMPLE

    # run ICA
    print "running ica"
    nfeatures = len(dfs)
    ica = FastICA(nfeatures)
    ica.fit(data)
   
    print "plotting mixing matrix"
    mm = ica.get_mixing_matrix()
    votes = np.sum(np.abs(mm) > (1. / float(nfeatures)), 0)
    print "channel votes", votes
    bcs = np.where(votes > 8)[0]
    mm[:,bcs] = np.zeros_like(mm[:,bcs])
    try:
        umm = np.linalg.inv(mm)
    except np.linalg.LinAlgError:
        print "Failed to mathematically invert matrix, trying pseudo-inverse"
        umm = np.linalg.pinv(mm)
    umm = np.matrix(umm)
    #ica.unmixing_matrix_ = umm
    #pl.figure()
    #pl.imshow(mm, interpolation='none')
    #pl.colorbar()
    print "mixing matrix shape:", mm.shape
    # mm is CHANNELS x COMPONENTS, so :,0 is all channels component 0
   
    print "reading all data"
    for df in dfs:
        df.seek(0)
    n = dfs[0].nframes
    fdata = [df.read_frames(n) for df in dfs]
    print "transformaing data"
    S = ica.transform(fdata)
    print "cleaning"
    cdata = np.array(umm * S)
    print "saving"
    oafs = ["%s/%s_%s" % (os.path.dirname(af), "clean", os.path.basename(af)) for af in afs]
    for (i, oaf) in enumerate(oafs):
        output = al.Sndfile(oaf, 'w', dfs[0].format, 1, dfs[0].samplerate)
        output.write_frames(cdata[i])
        output.sync()
        output.close()
    # test components
    #print "getting components"
    #pl.figure()
    #S = ica.transform(data)
    #NC = S.shape[0]
    #for ci in xrange(NC):
    #    #pl.subplot(NC,1,ci+1)
    #    pl.plot(S[ci,:]+ci*0.1)
    #    #print "%i : %f" % (ci, hurst(S[ci,:]))
    #
    #pl.xlim(0,4410)
    #
    # recombine
    #pl.show()

