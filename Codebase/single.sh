#!/bin/bash

#./$1.py $2 $3 $4
singularity exec /nfshome/singularity-images/csci4850-2022-Fall.sif ./$1.py $2 $3 $4

# $1 is the program
# $2 is the corpus 
# $3 is the training set
# $4 is the testing set

#echo './'$1'.py' $2 $3 $4
