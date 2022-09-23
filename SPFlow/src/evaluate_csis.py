from typing import Tuple
from joblib import load
from argparse import ArgumentParser
import pandas as pd
from torch import equal, index_put
from csitree import format_rule, CSITree
from dataset import Dataset, get_dataset
import boolean
import re
from itertools import combinations, product
import ast




def to_condition(x):
    negation = r'~([A-Za-z0-9]+)'
    term = r'([A-Za-z0-9]+)'
    
    return re.sub(term, r'(\1 == 1)', x)

parser = ArgumentParser()
parser.add_argument('dataset')
parser.add_argument('csi_path')
parser.add_argument('ground_truth_path')
args = parser.parse_args()

train, _ = get_dataset(args.dataset)

estimated = pd.read_csv(args.csi_path, converters={'C': ast.literal_eval})
gt = pd.read_csv(args.ground_truth_path, converters={'C': ast.literal_eval})
gt.A = gt.A.apply(to_condition)

df = pd.DataFrame(train.instances, columns = train.names)

import numpy as np
from functools import reduce
def score(x):
    subset = df.query(x.A)
    terms = gt[gt.C == x.C].A
    results = [subset.query(term).index for term in terms]
    if len(results) == 0:
        return 0.0
    i = np.unique(np.concatenate(results, axis=0)).tolist()
    return len(i) / len(subset)
        
estimated['ACC'] = estimated.apply(score, axis=1)

estimated = estimated.drop(['Valid', 'MP', 'MR', 'NI'], axis=1)

# print (len(estimated), len(estimated[estimated.ACC > 0.8]))
estimated.to_csv(f'{args.dataset}_eval.csv', index = False)
