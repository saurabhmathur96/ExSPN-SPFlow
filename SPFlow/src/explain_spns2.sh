#!/usr/bin/env bash
set -x # echo on

python exspn.py models/precomputed/asia.joblib asia explanations/precomputed/asia.joblib --precomputed-instance-function > timelogs.txt
python exspn.py models/precomputed/cancer.joblib cancer explanations/precomputed/cancer.joblib --precomputed-instance-function > timelogs.txt
python exspn.py models/precomputed/earthquake.joblib earthquake explanations/precomputed/earthquake.joblib --precomputed-instance-function > timelogs.txt
