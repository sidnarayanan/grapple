#!/bin/bash

#for dr in 0.0 0.2 0.4 0.6 1.0; do
for dr in 1.0 6.0; do
    ../train_pumet.py -c dr_${dr}.yaml &> log.${dr}
done 
