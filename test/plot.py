#!/usr/bin/env python

import glob

import icapp

import numpy
from pylab import *

mm, umm, cm, fns, c, t = icapp.fio.load_ica('ica.p')

ofns = sorted(glob.glob('data/*.wav'))
nfns = sorted(glob.glob('cleaned/*.wav'))

o = icapp.fio.MultiAudioFile(ofns)
n = icapp.fio.MultiAudioFile(nfns)

od = o.read(441)
nd = n.read(441)

figure(1)
subplot(311)
[plot(od[i] + i) for i in xrange(od.shape[0])]
ylabel("Raw")
subplot(312)
[plot(nd[i] + i) for i in xrange(nd.shape[0])]
ylabel("Clean")
#imshow(vstack((od, nd)))
#yticks([16, 48], ["old", "new"])
#plot(od.T)
#subplot(322)
#plot(nd.T)

imkwargs = {'interpolation': 'nearest'}
subplot(325)
imshow(mm, **imkwargs)
colorbar()
title("Mixing")
subplot(326)
imshow(umm, **imkwargs)
colorbar()
title("Un-mixing")

show()
