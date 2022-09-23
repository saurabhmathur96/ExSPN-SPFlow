#!/usr/bin/env bash
set -x # echo on
datasets=( artificial mushroom plants nltcs msnbc abalone adult wine car yeast ) # numom2b
for name in "${datasets[@]}"
do
    time python train.py $name "models/precomputed/${name}.joblib" --precomputed-instance-function > timelogs.txt
done

#for name in "${datasets[@]}"
#do
#    time python train.py $name "models/not-precomputed/${name}.joblib" 
#done
