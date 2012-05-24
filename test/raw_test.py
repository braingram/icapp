#!/usr/bin/env python

import numpy
import pylab
from scikits.learn.decomposition import FastICA

# make 5 fake signals with 3 components
sf = 5000
nfreq = 10
comps = [850, 90, nfreq]  # hz
chans = [
        [1.0, 0.0, 1.0],
        [0.7, 0.5, 1.0],
        [0.3, 0.1, 1.3],
        [0.0, 0.5, 1.7],
        [0.1, 0.1, 1.0],
        ]
nlvl = 0.0
ns = int(.5 * sf)
t = numpy.arange(ns) / float(sf)

components = numpy.array([]).reshape((0, ns))
for co in comps:
    s = numpy.sin(2. * numpy.pi * co * t)
    components = numpy.vstack((components, s))
print "Components:", components.shape

channels = numpy.array([]).reshape((0, ns))
for ch in chans:
    channel = numpy.random.randn(ns) * nlvl
    for (co, sub) in zip(components, ch):
        channel += co * sub
    print "stacking"
    channels = numpy.vstack((channels, channel))
print "Channels:", channels.shape

# ica decomposition

#ica = FastICA(whiten=True)
ica = FastICA(len(comps))
ica.fit(channels)
f_components = ica.transform(channels)
mm = numpy.matrix(ica.get_mixing_matrix())
um = numpy.matrix(ica.unmixing_matrix_)

# ======= Notes ========
# mixing matrix shape:
#   (channels, components)
#
# unmixing matrix shape:
#   (components, channels)
#
# to remove component i
# mm[:,i] = 0
# ======================

c_mm = mm.copy()
c_um = um.copy()

# try to find the nfreq hz component
closest = -1
dist = 1000000
freqs = numpy.fft.fftfreq(ns, d=1 / float(sf))
for (i, f_c) in enumerate(f_components):
    f = numpy.fft.fft(f_c)
    mf = numpy.abs(f[:len(f) / 2]).argmax()
    #freqs = numpy.fft.fftfreq(f.size, d=1 / float(sf))
    df = numpy.abs(freqs[mf] - nfreq)
    if df < dist:
        closest = i
        dist = df
    print "Component %s frequency %s" % (i, mf)

if closest != -1:
    print "Removing component %s" % closest
    i = closest
    print "Clearing", i
    v = c_mm[:, i]
    c_mm[:, i] = 0
    #c_um[closest, :] = 0
else:
    raise Exception("Noise not found!!!")

clean_matrix = c_mm * c_um
#c_channels = numpy.array(c_mm * c_um * channels)
c_channels = numpy.array(clean_matrix * channels)
print "Cleaned Channels", c_channels.shape


# plotting

pylab.subplot(331)
[pylab.plot(t, c) for c in components]
pylab.title("Components")

pylab.subplot(334)
[pylab.plot(t, c) for c in f_components]
pylab.title("Fit Components")

pylab.subplot(332)
[pylab.plot(t, c) for c in channels]
pylab.title("Channels")

pylab.subplot(337)
pylab.imshow(pylab.vstack((mm, c_mm)), interpolation='nearest')
pylab.title("Mixing Matrices")

pylab.subplot(338)
pylab.imshow(pylab.vstack((um, c_um)), interpolation='nearest')
pylab.title("Un-mixing Matrix")

pylab.subplot(339)
pylab.imshow(clean_matrix, interpolation='nearest')
pylab.title("Cleaning Matrix")

#pylab.subplot(336)
#pylab.imshow(c_mm, interpolation='nearest')
#pylab.title("Cleaned Mixing Matrix")

pylab.subplot(335)
[pylab.plot(t, c) for c in c_channels]
pylab.title("Cleaned channels")

pylab.subplot(333)
[pylab.psd(c, Fs=sf) for c in channels]
pylab.title("Channel PSDs")

pylab.subplot(336)
[pylab.psd(c, Fs=sf) for c in c_channels]
pylab.title("Cleaned Channel PSDs")

pylab.tight_layout()

pylab.show()
