from sklearn.tree import _tree
from sklearn.tree import DecisionTreeClassifier
from spn.structure.Base import Sum, Product, Leaf, get_nodes_by_type, get_topological_order
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

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
        node.independences = [[names[j] for j in child.scope] for child in node.children]
    
    sum_nodes = get_nodes_by_type(net, (Sum))
    for node in sum_nodes:
        node.scope_names = [names[j] for j in node.scope]
        node.n_instances = len(node.y)


        X, y = node.X, node.y
        XX = pd.DataFrame({str(j): X[:, i] if not categorical[j] else X[:, i].astype(int)
                        for i, j in enumerate(node.scope)})
        XX = pd.get_dummies(XX, 
                            columns = [str(j) for j in node.scope if categorical[j]], 
                            prefix_sep = "=")

        rules = []
        threshold = min_impurity_decrease
        tries = 5
        while len(rules) < 2 and tries >= 0:
            dt = DecisionTreeClassifier(min_impurity_decrease = threshold, 
                                        max_depth = max_depth,
                                        class_weight = "balanced")
            dt.fit(XX, y.astype(int))
            matrix = confusion_matrix(y, dt.predict(XX))
            
            
            r_scores = matrix.diagonal() / matrix.sum(axis=1)
            p_scores = matrix.diagonal() / matrix.sum(axis=0)
            features = ["%s=%s" % (names[int(c.split("=")[0])], c.split("=")[1]) if "=" in c 
                                    else names[int(c)]
                                    for c in XX.columns]
            rules = extract_rules(dt, features, "=")
            if tries == 1:
                threshold = 0
            else:
                tries -= 1
                threshold *= 0.5
            
        
        groups = pd.DataFrame(rules, columns=["rule", "cluster"]).groupby(by="cluster")
        for cluster, group in groups:
            node.children[cluster].condition  = group.rule.tolist()
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
                child.conditions_are_valid = child.conditions_are_valid and (len(child.condition) > 0)
                
                # sum -> node
                child.conditions.append(child.condition)
                child.min_recall = min(child.min_recall, child.recall)
                child.min_precision = min(child.min_precision, child.precision)
            

def context_specific_independences(net, precision_threshold = 0 , recall_threshold = 0, instance_threshold = 0):
    product_nodes = get_nodes_by_type(net, (Product))
    for node in product_nodes:
        if not node.has_condition:
            continue
        if not node.conditions_are_valid:
            continue
        
        filter_conditions = [(node.min_precision > precision_threshold),
                             (node.min_recall > recall_threshold),
                            (node.n_instances > instance_threshold)]
        if all(filter_conditions):
            yield (node.conditions, node.independences, node.min_precision, node.min_recall, node.n_instances)
            

def tree_edges(net, precision_threshold, recall_threshold, instance_threshold):
    nodes = get_nodes_by_type(net, (Sum, Product))
    for node in nodes:
        filter_conditions = [(node.conditions_are_valid),
                             (node.min_precision > precision_threshold),
                             (node.min_recall > recall_threshold),
                             (node.n_instances > instance_threshold)]
        if not all(filter_conditions):
            continue

        if isinstance(node, Sum):
            yield (node.conditions, node.scope_names, node.min_precision, node.min_recall, node.n_instances)
        else:
            yield (node.conditions, node.independences, node.min_precision, node.min_recall, node.n_instances)





def format_clause(clause, format_condition):
    subclauses = clause.split(" and ")
    if len(subclauses) == 1:
        return format_condition(subclauses[0])
    
    formatted_subclauses = ["(%s)" % format_condition(subclause) for subclause in subclauses]
    return " & ".join(formatted_subclauses)

def format_node(node, format_condition):
    if len(node) == 1:
        return format_clause(node[0], format_condition)
    
    formatted_clauses = ["(%s)" % format_clause(clause, format_condition) for clause in node]
    
    return ' | '.join(formatted_clauses)


def format_antecedent(antecedent, format_condition):
    if len(antecedent) == 1:
        return format_node(antecedent[0], format_condition)
    
    formatted_nodes = ["[%s]" % format_node(node, format_condition) for node in antecedent]
    return ' & '.join(formatted_nodes)

def format_consequent(consequent):
    cnodes = ["(%s)" % ",".join(node) for node in consequent]
    return ", ".join(cnodes)

def antecedent_count(antecedent):
    return len([subterm for term in antecedent.split('&') 
                                 for subterm in term.split('|')])

def consequent_count(consequent):
    return len(consequent.split(", "))


def build_instance_function(net, data):
    """ add X and y attributes to sum nodes """
    pass