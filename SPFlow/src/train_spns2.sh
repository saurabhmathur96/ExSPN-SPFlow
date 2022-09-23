#!/usr/bin/env bash
set -x # echo on
python train.py asia "models/precomputed/asia.joblib" --precomputed-instance-function > timelogs1.txt
python train.py cancer "models/precomputed/cancer.joblib" --precomputed-instance-function > timelogs2.txt
python train.py earthquake "models/precomputed/earthquake.joblib" --precomputed-instance-function > timelogs3.txt