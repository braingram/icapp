#!/usr/bin/env python

import os

import numpy
from pylab import *

from scikits.audiolab import Sndfile as SF
from scikits.audiolab import Format

dd = 'data/'
nsamps = 44100  # 1 second
srate = 44100
mode = 'pcm24'
nchan = 4
schan = 1
nlvl = 0.1
to_fn = lambda i: "%s/input_%i#01.wav" % (dd, i)

t = arange(nsamps, dtype='f8') / float(srate)
# 1500 hs sin wave on ch 1, 2, 3, 4
noise = sin(2 * pi * t * 2500) * 0.5
# 1000 hz sin wave on ch 1
signal = sin(2 * pi * t * 1000) * 0.9

if not os.path.exists(dd):
    os.makedirs(dd)

fformat = Format(encoding=mode)
for i in xrange(1, nchan + 1):
    f = SF(to_fn(i), 'w', fformat, 1, srate)
    if i == schan:
        f.write_frames(noise + signal + randn(len(t)) * nlvl)
    else:
        f.write_frames(noise + randn(len(t)) * nlvl)
    f.sync()
    f.close()
