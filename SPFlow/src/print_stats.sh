#!/usr/bin/env bash
datasets=( artificial mushroom plants nltcs msnbc abalone adult wine car yeast numom2b )
for name in "${datasets[@]}"
do
    python csi_stats.py explanations/precomputed/${name}.joblib $name 
done

for name in "${datasets[@]}"
do
    python csi_stats.py explanations/not-precomputed/${name}.joblib $name 
done