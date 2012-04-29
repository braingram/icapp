#!/bin/bash

ARGS="-v -m simple -s 4100 -c 2"

# clean up
rm -rf data cleaned cleaned ica.p

echo "making data"
python make_data.py
sync

FNS=`find data/ -name *.wav | sort`

# run new ica
echo "running ica"
ica_clean.py -CO 'cleaned/' $ARGS $FNS
sync

# compare results
echo "plotting results"
python plot.py

# clean up
rm -rf data cleaned cleaned ica.p
