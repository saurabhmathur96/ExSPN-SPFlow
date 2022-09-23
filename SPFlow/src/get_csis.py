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

parser = ArgumentParser()
parser.add_argument('csitree_path')
parser.add_argument('dataset_name')
args = parser.parse_args()

csi: CSITree = load(args.csitree_path)
train, _ = get_dataset(args.dataset_name)
na_rules = 0
antecedent_counts = []
consequent_counts = []
rules = [(*format_rule(rule, train), *rule[2:]) for rule in csi.rules()]
filtered_rules = [rule for rule in rules if ('N/A' not in rule[0]) and rule[-1]]

algebra = boolean.BooleanAlgebra()
df = pd.DataFrame(rules, columns = ['A', 'C', 'MP', 'MR', 'NI', 'Valid'])


if args.dataset_name == 'artificial':
    min_instances = len(train.instances) / 5
elif args.dataset_name == 'numom2b':
    min_instances = sum(train.instances[:, -1]) / 100
else:
    min_instances = len(train.instances) / 100

df = df[(df.MP > 0.7) & (df.MR > 0.7) & (df.NI > min(5 * min_instances, len(train.instances) // 5))]

equality = r'([A-Za-z0-9]+) == ([0-2])'
inequality = r'([A-Za-z0-9]+) != ([0-2])'

def to_ascii(x):
    x = re.sub(equality, r'\1_\2',  re.sub(inequality, r'~\1_\2', x))
    x = x.replace('∧', '&').replace('∨', '|').replace('¬', '~')
    return x

def to_condition(x):
    negation = r'~([A-Za-z0-9]+)'
    term = r'([A-Za-z0-9]+)'
    
    return re.sub(term, r'(\1 == 1)', x)

def simplify(x):
    parsed = algebra.parse(x, simplify = False)
    return str(parsed)

df.A = df.A.apply(to_ascii) \
            .apply(algebra.parse) \
            .apply(str) 
#            .apply(to_condition)


def pairwise_independences(x):
    independences = []
    groups = x.lstrip('[').rstrip(']').split(' | ')
    for first, second in combinations(groups, r = 2):
        for each in product(first.split(','), second.split(',')):
            independences.append(tuple(sorted(each)))
    return list(set(independences))


df.C = df.C.apply(pairwise_independences)

import operator
from functools import reduce
df = df.explode('C').reset_index(drop = True)

def vaccuous(x):
    return any(v in str(x.A) for v in x.C)

df = df[~df.apply(vaccuous, axis=1)]

df.A = df.A.apply(to_condition)


df.to_csv('%s_csi.csv' % args.dataset_name, index = False)
