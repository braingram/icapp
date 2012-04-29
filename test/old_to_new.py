#!/usr/bin/env python

import os
import sys

import numpy

import icapp

fns = [os.path.basename(f) for f in sys.argv[1:]]
assert len(fns) == 32

mm = numpy.loadtxt('old_cleaned/mixingmatrix')
um = numpy.loadtxt('old_cleaned/unmixingmatrix')

cm = icapp.ica.make_cleaning_matrix(mm, um)

with open('old_cleaned/mixingmatrix') as F:
    for line in F:
        if 'threshold' in line:
            threshold = float(line.split()[2])
        elif 'count' in line:
            count = int(line.split()[2])


icapp.fio.save_ica('ica.p', mm, um, cm, fns, count, threshold)
