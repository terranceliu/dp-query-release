#!/bin/bash

# DC: 11
ALL_FIPS=(01 02 04 05 06 08 09 10 12 13 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 44 45 46 47 48 49 50 51 53 54 55 56 11)

for STATE_ID in "${ALL_FIPS[@]}"
do
  python examples/data_preprocessing/preprocess_ppmf.py --stateid $STATE_ID
done

TRACTS = (42003140100 42029302101 42071011804)