import pandas as pd
from sklearn.metrics import confusion_matrix
from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type, get_topological_order
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import _tree
from dataset import get_dataset, DataType
from spn.structure.Base import Sum, Product, Leaf
from spn.structure.Base import get_nodes_by_type, get_topological_order
from spn.algorithms.Inference import log_likelihood
from itertools import chain
from argparse import ArgumentParser
from csitree import CSITree, format_rule
from joblib import load, dump
import numpy as np
np.random.seed(0)


def extract_rules(dt, features, prefix_sep="_"):
    """ extract rules from a DecisionTreeClassifier"""
    rules = []

    def recurse(tree, i, partial):
        f = tree.feature[i]
        if f == _tree.TREE_UNDEFINED:

            if len(partial) > 1:
                antecedent = ' ^ '.join(['(%s)' % each for each in partial])
            else:
                antecedent = ''.join(partial)
            consequent = dt.classes_[tree.value[i].ravel().argmax()]
            rules.append((antecedent, consequent))
        else:
            name = features[f]
            if prefix_sep in name:
                feature, threshold = name.split(prefix_sep)
                rule_left = "%s != %s" % (feature, threshold)
                rule_right = "%s == %s" % (feature, threshold)
            else:
                feature = name
                threshold = tree.threshold[i]
                rule_left = "%s <= %.4f" % (feature, threshold)
                rule_right = "%s > %.4f" % (feature, threshold)

            left = tree.children_left[i]
            if left >= 0:
                recurse(tree, left, partial + [rule_left])

            right = tree.children_right[i]
            if right >= 0:
                recurse(tree, right, partial + [rule_right])
    recurse(dt.tree_, 0, [])
    return rules


def annotate_nodes(net, names, categorical, min_impurity_decrease, max_depth):
    """ approximates sum nodes as rules and annotates product nodes with independences"""
    for node in get_nodes_by_type(net, (Sum, Product)):
        node.has_condition = False

    product_nodes = get_nodes_by_type(net, (Product))
    for node in product_nodes:
        node.independences = [[names[j] for j in child.scope]
                              for child in node.children]

    sum_nodes = get_nodes_by_type(net, (Sum))
    for node in sum_nodes:
        node.scope_names = [names[j] for j in node.scope]
        node.n_instances = len(node.y)
        if node.n_instances == 0:
            for child in node.children:
                child.n_instances = 0
                child.precision = 0
                child.recall = 0
                child.has_condition = False

            continue

        X, y = node.X, node.y
        #from collections import Counter
        #print (node.scope, X.dtype, X.shape)
        #print (Counter(X[:, -1]))
        # print (node.X.shape)
        XX = pd.DataFrame({str(j): X[:, i] if not categorical[j] else X[:, i].astype(int)
                           for i, j in enumerate(node.scope)})
        #if 22 in node.scope:
        #    print (XX['22'].unique())
        XX = pd.get_dummies(XX,
                            columns=[str(j)
                                     for j in node.scope if categorical[j]],
                            prefix_sep="=")
        
        # print ([e for e in XX.columns if "22" in e])
        rules = []
        params = dict(threshold = min_impurity_decrease, max_depth = max_depth)
        
        tries = 10
        while len(rules) < 2 and tries >= 0:
            dt = DecisionTreeClassifier(min_impurity_decrease=params['threshold'],
                                        max_depth=params['max_depth'],
                                        class_weight="balanced")
            dt.fit(XX, y.astype(int))
            matrix = confusion_matrix(y, dt.predict(XX), labels = [0, 1])

            r_scores = matrix.diagonal() / matrix.sum(axis=1)
            p_scores = matrix.diagonal() / matrix.sum(axis=0)
            features = ["%s=%s" % (names[int(c.split("=")[0])], c.split("=")[1]) if "=" in c
                        else names[int(c)]
                        for c in XX.columns]
            # print ([f for f in features if "class" in f])
            rules = extract_rules(dt, features, "=")
            if tries == 1:
                params['threshold'] = 0
                params['max_depth'] = 1
            
            else:
                params['threshold'] *= 0.5
            tries -= 1
        # print (r_scores)
        # print (p_scores)
        groups = pd.DataFrame(
            rules, columns=["rule", "cluster"]).groupby(by="cluster")
        for child in node.children:
            child.n_instances = 0
            child.precision = 0
            child.recall = 0
        for cluster, group in groups:
            node.children[cluster].condition = group.rule.tolist()
            node.children[cluster].has_condition = True
            node.children[cluster].recall = r_scores[cluster]
            node.children[cluster].precision = p_scores[cluster]
            node.children[cluster].n_instances = sum(y == cluster)


def annotate_spn(net, names, categorical, min_impurity_decrease=0.1, max_depth=1):
    """ annotate each node with all conditions from its ancestors """
    annotate_nodes(net, names, categorical, min_impurity_decrease, max_depth)

    net.conditions = []
    net.precision = 1.0
    net.recall = 1.0
    if isinstance(net, Sum):
        net.n_instances = len(net.y)
    else:
        net.n_instances = 0
        for child in net.children:
            if isinstance(net, Sum):
                net.n_instances = len(child.y)
                break
    net.min_recall = 1.0
    net.min_precision = 1.0
    net.has_condition = True
    net.conditions_are_valid = True

    for node in get_topological_order(net)[::-1]:
        if isinstance(node, Leaf):
            continue

        for child in node.children:
            if isinstance(child, Leaf):
                continue

            # sum/product -> node
            child.conditions = list(node.conditions)
            child.min_recall = node.min_recall
            child.min_precision = node.min_precision
            child.conditions_are_valid = node.conditions_are_valid

            if child.has_condition:
                child.conditions_are_valid = child.conditions_are_valid and (
                    len(child.condition) > 0)

                # sum -> node
                child.conditions.append(child.condition)
                child.min_recall = min(child.min_recall, child.recall)
                child.min_precision = min(child.min_precision, child.precision)


def build_instance_function(net, data):
    net.data = np.array(data.instances)
    for node in get_topological_order(net)[::-1]:
        if type(node) == Leaf:
            continue

        if type(node) == Product:
            for child in node.children:
                child.data = np.array(node.data)
        elif type(node) == Sum:
            likelihoods = np.concatenate(
                [log_likelihood(child, node.data) for child in node.children], axis=1)
            y = likelihoods.argmax(axis=1)

            for i, child in enumerate(node.children):
                child.data = np.array(node.data[y == i])

    def get_instances(node):
        return node.data
    return net, get_instances


def remove_prefix(ls, prefix):
    return ls[len(prefix):]


def explain(net, names, data_types, data=None):
    if data is not None:
        net, get_instances = build_instance_function(net, data)
        for n in get_nodes_by_type(net, (Sum)):
            n.y = np.concatenate([np.zeros(len(get_instances(child))) + j
                                  for j, child in enumerate(n.children)])
            n.X = np.concatenate([get_instances(child)[:, n.scope]
                                  for child in n.children])

    else:
        for n in get_nodes_by_type(net, (Sum)):
            assert hasattr(n, 'X')
            assert hasattr(n, 'y')

    annotate_spn(
        net, names, [t in (DataType.CATEGORICAL, DataType.BINARY) for t in data_types], max_depth=2)

    def recurse(node, prefix=None):
        assert hasattr(node, 'children')

        if prefix is None:
            prefix = []

        conditions = remove_prefix(node.conditions, prefix)
        def is_product(n): return (type(n) == Product)
        def is_sum(n): return (type(n) == Sum)
        def is_leaf(n): return (isinstance(n, Leaf))

        def has_leaf_children(n): return all(is_leaf(child)
                                             for child in n.children)

        if is_product(node):
            if has_leaf_children(node):
                return CSITree(node.name, conditions, node.conditions, node.independences, [],
                               node.min_precision, node.min_recall, node.n_instances, condition_valid = node.has_condition)

        # sum node
        children = []
        blocks = []
        for child in node.children:

            if is_leaf(child):
                children.append(None)

            elif is_sum(child):
                # product/sum -> sum edge
                gchildren = list(child.children)
                while any(is_sum(each) for each in gchildren):
                    gc = [each.children if is_sum(each) else [each]
                          for each in gchildren]
                    gchildren = list(chain.from_iterable(gc))
                children.append([recurse(child, node.conditions)
                                 for child in gchildren])
            else:
                # sum/product -> product edge
                children.append([recurse(child, node.conditions)])

        if is_product(node):
            blocks = [[names[j] for j in child.scope]
                      for child in node.children]

        else:
            blocks = [[names[j] for j in node.scope]]

        return CSITree(node.name, conditions, node.conditions, blocks, children,
                       node.min_precision, node.min_recall, node.n_instances, condition_valid = node.has_condition)

    return recurse(net)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('spn_path')
    parser.add_argument('dataset_name')
    parser.add_argument('save_path')
    parser.add_argument('--precomputed-instance-function',
                        dest='precomputed', action='store_true', default=False)
    args = parser.parse_args()

    net = load(args.spn_path)
    train, test = get_dataset(args.dataset_name)

    if not args.precomputed:
        print ("not precomputed")
        csi = explain(net, train.names, train.data_types, train)
    else:
        print ("precomputed")
        for each in get_nodes_by_type(net, Sum):
            print (each.X.shape)
        csi = explain(net, train.names, train.data_types, None)
    
    for rule in csi.rules():
        print ("%s => %s" % format_rule(rule, train)) 

    dump(csi, args.save_path)
