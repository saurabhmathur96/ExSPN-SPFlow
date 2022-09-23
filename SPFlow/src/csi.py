from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def extract_rules(dt, features, prefix_sep="_"):
    """ extract rules from a DecisionTreeClassifier"""
    rules = []
    def recurse(tree, i, partial):
        f = tree.feature[i]
        if f == _tree.TREE_UNDEFINED:
            antecedent = " and ".join(partial)
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
                rule_left = "%s <= %s" % (feature, threshold)
                rule_right = "%s > %s" % (feature, threshold)

            left = tree.children_left[i]
            if left >= 0:
                recurse(tree, left, partial + [rule_left])

            right = tree.children_right[i]        
            if right >= 0:
                recurse(tree, right, partial + [rule_right])
    recurse(dt.tree_, 0, [])
    return rules

def annotate_nodes(net, names, categorical, min_impurity_decrease, min_instances_slice = 20):
    """ approximates sum nodes as rules and annotates product nodes with independences"""
    product_nodes = get_nodes_by_type(net, (Product))
    for node in product_nodes:
        node.independences = [[names[j] for j in child.scope] for child in node.children]
        node.valid = False

    sum_nodes = get_nodes_by_type(net, (Sum))
    for node in sum_nodes:
        X, y = node.X, node.y
        XX = pd.DataFrame({str(j): X[:, i] if not categorical[j] else X[:, i].astype(int)
                           for i, j in enumerate(node.scope)})
        XX = pd.get_dummies(XX, columns=[str(j) for j in node.scope if categorical[j]], prefix_sep="=")
        dt = DecisionTreeClassifier(min_impurity_decrease=min_impurity_decrease, class_weight="balanced")
        #max_leaf_nodes = len(node.children))
        # dt = DecisionTreeClassifier(max_depth = 2)
        dt.fit(XX, y.astype(int))
        matrix = confusion_matrix(y, dt.predict(XX))
        scores = matrix.diagonal() / matrix.sum(axis=1)
        
        features = [names[int(c)] if "=" not in c else "%s=%s" % (names[int(c.split("=")[0])], c.split("=")[1]) for c in XX.columns]
        rules = extract_rules(dt, features, "=")
        groups = pd.DataFrame(rules, columns=["rule", "cluster"]).groupby(by="cluster")
        for i, score in enumerate(scores):
            node.children[i].score = score
            node.children[i].valid = False
            node.children[i].support = sum(y == i)
            
        for cluster, group in groups:
            node.features = features
            node.dt = dt
            node.children[cluster].condition = group.rule.tolist()
            node.children[cluster].support = matrix.diagonal()[cluster]
            node.children[cluster].valid = (node.children[cluster].support > min_instances_slice)
        

def annotate(net, names, categorical, min_impurity_decrease=0.1, min_instances_slice=20):
    """ annotate each node with all conditions from its ancestors """
    def recurse(node, conditions, agg):
        if node is None or isinstance(node, Leaf):
            return
        
        if hasattr(node, 'valid') and not node.valid:
            return
        
        if hasattr(node, 'condition'): 
            # print (repr( ''.join(node.condition)))
            if not ''.join(node.condition):
                # print (repr(''.join(node.condition)))
                # print (conditions)
                return
            
            conditions += [node.condition]
            agg *= node.score
            
        node.conditions = conditions
        node.agg = agg
        
        for child in node.children:
            recurse(child, list(conditions), agg)
    
    annotate_nodes(net, names, categorical, min_impurity_decrease, min_instances_slice)
    net.conditions = []
    net.agg = 1.0
    if isinstance(net, Product):
        net.valid = True
        net.support = np.inf
    recurse(net, [], 1.0)
    

def context_specific_independences(net):
    product_nodes = get_nodes_by_type(net, (Product))
    for node in product_nodes:
        if hasattr(node, 'valid') and node.valid and hasattr(node, 'conditions'):
            yield (node.conditions, node.independences, node.agg, node.support)

def tree_edges(net, names):
    nodes = get_nodes_by_type(net, (Sum, Product))
    for node in nodes:
        if hasattr(node, 'valid') and node.valid and hasattr(node, 'conditions'):
            if isinstance(node, Product):
                yield (node.conditions, node.independences, node.agg, node.support)
            else:
                independences = [[names[j] for j in node.scope]]
                yield (node.conditions, independences, node.agg, node.support)
