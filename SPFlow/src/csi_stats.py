from typing import Tuple
from joblib import load
from argparse import ArgumentParser
from csitree import format_rule, CSITree
from dataset import Dataset, get_dataset

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
acount = lambda a: len([subterm for term in a.split('∧') for subterm in term.split('∨')])
ccount = lambda c: len(c.split('|'))


if args.dataset_name == 'artificial':
    min_instances = len(train.instances) / 5
elif args.dataset_name == 'numom2b':
    min_instances = sum(train.instances[:, -1]) / 100
else:
    min_instances = len(train.instances) / 100

import pandas as pd
frame = pd.DataFrame([(acount(a), ccount(c), mp, mr, ni) for a, c, mp, mr, ni, _ in filtered_rules], 
                      columns=['acount', 'ccount', 'mp', 'mr', 'ni'])

frame2 = frame[(frame.mp > 0.7) & (frame.mp > 0.7) & (frame.ni > min(5 * min_instances, len(train.instances) // 5))]

fmt = '%s, %d, %.2f, %.2f, %d, %.2f, %.2f' 
params = (args.dataset_name, len(frame), frame.acount.mean(), 
          frame.ccount.mean(), len(frame2), frame2.acount.mean(), 
          frame2.ccount.mean())

print (fmt % params)
#print (len(rules))
#print ('%.4f' % (sum(antecedent_counts) / len(rules)))
#print ('%.4f' % (sum(consequent_counts) / len(rules)))