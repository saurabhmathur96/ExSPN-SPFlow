from sklearn.datasets import fetch_openml
from collections import namedtuple
import enum
from pickletools import read_decimalnl_long
import numpy as np
import pandas as pd
from sympy import is_nthpow_residue
from tqdm import tqdm
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.leaves.parametric.Parametric import Bernoulli, Categorical, Gaussian
from spn.structure.Base import Context, Sum, Product, Leaf
from spn.structure.Base import get_nodes_by_type, get_topological_order
from spn.algorithms.Inference import log_likelihood
import csi2
from sklearn.model_selection import train_test_split
from itertools import chain


class DataType(enum.Enum):
    CONTINUOUS = 1
    BINARY = 2
    CATEGORICAL = 3


Dataset = namedtuple('Dataset', ['names',
                                 'data_types',
                                 'categories',
                                 'instances'])


def get_dataset(name):
    if name == "artificial":
        np.random.seed(0)
        N = 10000
        data1 = np.random.multivariate_normal(
            mean=[0, 2, 2, 2], cov=np.eye(4) * 0.01, size=N)
        cov = np.eye(4)
        cov[1, 2] = cov[2, 1] = 1
        cov[2, 3] = cov[3, 2] = 1
        cov[1, 3] = cov[3, 1] = 1
        data2 = np.random.multivariate_normal(
            mean=[-8, 4, 4, 4], cov=cov * 0.01, size=N)

        cov = np.eye(4)
        cov[2, 3] = cov[3, 2] = 1
        data3 = np.random.multivariate_normal(
            mean=[-8, 8, 4, 4], cov=cov * 0.01, size=N)

        data = np.concatenate([data1, data2, data3], axis=0)
        names = ['V%d' % i for i in range(4)]
        data_types = [DataType.CONTINUOUS for _ in names]


        train, test = train_test_split(data, random_state=0)
        print(len(train), len(test))

        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=None
        )
        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=None
        )

        return train_set, test_set
    elif name == "mushroom":
        bunch = fetch_openml('mushroom')

        frame = bunch.frame.dropna() # pd.read_csv("mushroom.csv")
        names = [name.replace("%3F", "") for name in frame.columns.tolist()]
        data_types = [DataType.CATEGORICAL for _ in names]
        cols = []
        value_text = []
        for name in frame.columns:
            values, text = pd.factorize(frame[name], sort=True)
            value_text.append(text.tolist())
            cols.append(values)
            print (values[0])
            print (values.shape)
        print (cols)
        data = np.stack(cols, axis=1)
        train_set, test_set = train_test_split(data, stratify = frame['class'], random_state = 0)
        categories = value_text #{i: each for i, each in enumerate(value_text)}

        
        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=categories
        )
        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=categories
        )

        
        return train_set, test_set
    elif name == "plants":
        state_names = set()
        for line in open('plants.data', encoding='ISO-8859-1'):
            name, *states = line.rstrip().split(",")
            state_names.update(states)

        features = list(sorted(list(state_names)))
        rows = []
        for line in open('plants.data', encoding='ISO-8859-1'):
            name, *states = line.rstrip().split(",")
            row = np.zeros(len(features))
            for state in states:
                row[features.index(state)] = 1
            rows.append(np.array(row))

        D = np.array(rows)
        D = D[D.sum(axis=1) >= 2]
        data_types = [DataType.BINARY for _ in features]

        train_set, test_set = train_test_split(D, random_state = 0)

        train_set = Dataset(
            names=features,
            data_types=data_types,
            instances=train,
            categories=None
        )

        test_set = Dataset(
            names=features,
            data_types=data_types,
            instances=test,
            categories=None
        )

        train_set, test_set
    elif name == "nltcs":
        names = ['telephoning', 'medicine', 'money', 'traveling', 'outside',
                 'grocery', 'cooking', 'laundry', 'light', 'heavy',
                 'toilet', 'bathing', 'dressing', 'inside', 'bed', 'eating']

        frame = pd.read_csv("nltcs/2to16disabilityNLTCS.txt",
                            sep='\t').drop(['PERCENT'], axis=1)
        frame = frame.rename({
            old: new for old, new in zip(frame.columns, names)
        }, axis=1)
        D = frame.to_numpy()
        expanded = []
        for row in D:
            expanded.append([list(row[0:-1])] * row[-1])
        D = np.concatenate(expanded, axis=0)
        frame = pd.DataFrame(D, columns=frame.columns[:-1])
        D = frame.to_numpy()
        names = frame.columns.tolist()
        data_types = [DataType.BINARY for _ in names]

        train, test = train_test_split(D, random_state = 0)

        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=None
        )
        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=None
        )

        return train_set, test_set

    elif name == "msnbc":
        f = open('msnbc.txt')
        next(f)
        next(f)
        names = next(f).rstrip().split()
        next(f)
        next(f)
        next(f)
        next(f)
        data = []
        for row in tqdm(f, total=989818):
            v = np.zeros(len(names), dtype="bool")
            row = row.rstrip().split(" ")
            for each in row:
                each = int(each) - 1
                v[each] = 1
            data.append(v)

        data = np.array(data)
        data = data[data.sum(axis=1) >= 2]
        data_types = [DataType.BINARY for _ in names]
        train, test = train_test_split(data, random_state = 0)

        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=None
        )
        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=None
        ) 

        return train_set, test_set

    elif name == "abalone":
        bunch = fetch_openml(data_id = 183)
        frame = bunch.frame
        frame.Class_number_of_rings = frame.Class_number_of_rings.astype(int)
        values, text = pd.factorize(frame.Sex, sort=True)
        frame.Sex = values

        names = frame.columns.tolist()
        data_types = [DataType.CATEGORICAL] + [DataType.CONTINUOUS for _ in names[1:]]
        train, test = train_test_split(frame.to_numpy(), random_state = 0)
        categories = [values] + ([None] * len(names[1:]))

        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=categories
        )

        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=categories
        )

        return train_set, test_set
    elif name == "adult":
        bunch = fetch_openml('adult', version = 2)

        frame = bunch.frame.dropna()
        names = frame.columns.tolist()
        continuous = ['age', 'fnlwgt', 'education-num', 
                    'capital-gain', 'capital-loss', 'hours-per-week']
        categories = [name for name in names if name not in continuous]
        data_types = [DataType.CATEGORICAL if name in categories 
        else DataType.CONTINUOUS for name in names]
        cols = []
        value_text = []
        for name in names:
            if name in categories:
                values, text = pd.factorize(frame[name], sort=True)
                value_text.append(text.tolist())
                cols.append(values)
            else:
                cols.append(frame[name])
                value_text.append(None)
            
        data = np.stack(cols, axis=1)

        train, test = train_test_split(data, stratify = frame['class'], random_state = 0)
        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=value_text
        )

        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=value_text
        )
        return train_set, test_set

    elif name == "wine":
        red = pd.read_csv("winequality-red.csv", sep=";")
        oldnames = red.columns
        rename_dict = {old: old.replace(" ", "_") for old in oldnames}
        red = red.rename(rename_dict, axis=1)

        white = pd.read_csv("winequality-white.csv", sep=";")
        oldnames = white.columns
        rename_dict = {old: old.replace(" ", "_") for old in oldnames}
        white = white.rename(rename_dict, axis=1)
        # print ("%d red wines and %d white wines" % (len(red), len(white)))

        frame = pd.concat([red, white], ignore_index=True)
        features = frame.columns.tolist()
        data_types =  [DataType.CONTINUOUS for _ in features]

        data = frame.to_numpy().astype(float)
        train, test = train_test_split(data, random_state = 0)
        train_set = Dataset(
            names=features,
            data_types=data_types,
            instances=train,
            categories=None
        )

        test_set = Dataset(
            names=features,
            data_types=data_types,
            instances=test,
            categories=None
        )

        return train_set, test_set

    elif name == "car":
        bunch = fetch_openml(data_id = 40975)
        frame = bunch.frame.dropna()
        names = frame.columns.tolist()
        categories = list(names)
        data_types = [DataType.CATEGORICAL for _ in names]
        cols = []
        value_text = []
        for name in names:
            if name in categories:
                values, text = pd.factorize(frame[name], sort=True)
                value_text.append(text.tolist())
                cols.append(values)
            else:
                cols.append(frame[name])
                value_text.append(None)

        data = np.stack(cols, axis=1)
        train, test = train_test_split(data, random_state = 0)

        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=value_text
        )

        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=test,
            categories=value_text
        )

        return train_set, test_set

    elif name == "yeast":
        bunch = fetch_openml(data_id = 181)
        frame = bunch.frame.dropna()
        names = frame.columns.tolist()
        continuous = 'mcg,gvh,alm,mit,erl,pox,vac,nuc'.split(",")
        categories = [name for name in names if name not in continuous]
        data_types = [DataType.CATEGORICAL if name in categories else DataType.CONTINUOUS for name in names]

        cols = []
        value_text = []
        for name in names:
            if name in categories:
                values, text = pd.factorize(frame[name], sort=True)
                value_text.append(text.tolist())
                cols.append(values)
            else:
                cols.append(frame[name])
                value_text.append(None)

        data = np.stack(cols, axis=1)
        train, test = train_test_split(data, stratify = frame.class_protein_localization, random_state = 0)
        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=value_text
        )
        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=value_text
        )
        return train_set, test_set
        
    elif name == "numom2b":
        frame = pd.read_excel("data_gdm_discrete_BMI_merged.xlsx", engine="openpyxl")
        frame.oDM = (frame.oDM == 2).astype(int)
        for name in frame.columns:
            frame[name] = frame[name].astype('category')
        data = frame.to_numpy()
        names = frame.columns.tolist()
        
        data_types = [DataType.CATEGORICAL for n in names]
        
        train, test = train_test_split(data, stratify = data[:, -1], random_state = 0)
        train_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=None
        )
        test_set = Dataset(
            names=names,
            data_types=data_types,
            instances=train,
            categories=None
        )
        return train_set, test_set
    else:
        raise ValueError('invalid name')


def remove_prefix(ls, prefix):
    return ls[len(prefix):]

def explain(net, names, data_types, data=None):
    if data is not None:
        net, get_instances = build_instance_function(net, data)
        for n in get_nodes_by_type(net, (Sum)):
            n.y = np.concatenate([np.zeros(len(get_instances(child))) + j
                                  for j, child in enumerate(n.children)])
            n.X = np.concatenate([get_instances(child)
                                  for child in n.children])

    else:
        for n in get_nodes_by_type(net, (Sum)):
            assert hasattr(n, 'X')
            assert hasattr(n, 'y')
    

    csi2.annotate_spn(
        net, names, [t in (DataType.CATEGORICAL, DataType.BINARY) for t in data_types], max_depth=2)

    def recurse(node, prefix = None):
        assert hasattr(node, 'children')

        if prefix is None:
            prefix = []

        conditions = remove_prefix(node.conditions, prefix)
        is_product = lambda n: (type(n) == Product)
        is_sum = lambda n: (type(n) == Sum)
        is_leaf = lambda n: (isinstance(n, Leaf))
        has_leaf_children = lambda n: all(is_leaf(child) for child in n.children)

        if is_product(node):
            if has_leaf_children(node):
                return CSITree(node.name, conditions, node.conditions, node.independences, [],
                node.min_precision, node.min_recall, node.n_instances)
        
        """
            blocks = [[names[j] for j in child.scope]
                      for child in node.children]
            children = [recurse(child) if type(child) == Sum else None
                        for child in node.children]
            return CSITree(node.name, node.conditions, blocks, children)

        else:
        """
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
                children.append([recurse(child, node.conditions) for child in gchildren])
            else:
                # sum/product -> product edge
                children.append([recurse(child, node.conditions)])
        
        if is_product(node):
            blocks = [[names[j] for j in child.scope] for child in node.children]

        else:
            blocks = [[names[j] for j in node.scope]]
        
        return CSITree(node.name, conditions, node.conditions, blocks, children, 
        node.min_precision, node.min_recall, node.n_instances)

    return recurse(net)




class CSITree:
    def __init__(self, name, conditions, expanded_conditions, blocks, children, mp, mr, n_instances):
        self.name = name
        self.expanded_conditions = expanded_conditions
        self.conditions = conditions
        self.blocks = blocks
        self.children = children
        self.mp = mp
        self.mr = mr
        self.n_instances = n_instances

    def to_dot(self):  # -> str
        # See https://www.ocf.berkeley.edu/~eek/index.html/tiny_examples/thinktank/src/gv1.7c/doc/dotguide.pdf
        nodes = [self] + [node for node in self]
        node2id = { node.name: i for i, node in enumerate(nodes)}

        lines = ["splines=false;", "node [shape = record,height=0.5];"]
        for i, node in enumerate(nodes):
            fields = ' | '.join(["<f%d> %s" % (j, ",".join(block)) for j, block in enumerate(node.blocks)])
            lines.append('node%d[label = "%s"];' % (i, fields))

        for (parent, i, child) in self.edges():
            parentid = node2id[parent.name]
            childid = node2id[child.name]
            
            lterms = []
            for term in child.conditions:
                if len(term) > 1:
                    term = ["(%s)" % st for st in term]
                partial = ' v '.join(term)
                if len(child.conditions) > 1:
                    partial = "(%s)" % partial
                lterms.append(partial)
            label = ' ^ '.join(lterms)
            line = '"node%d":f%d:s -> "node%d":f0:n [label="%s"];' % (parentid, i, childid, label)
            lines.append(line)
        
        return "digraph {\n%s\n}" % "\n".join(lines)
         
    def __repr__(self):
        blocks = str([block for block in self.blocks])
        return "Node(%s, %s)" % (self.name, blocks)

    def __iter__(self):
        for block in self.children:
            if block is None: continue
            for child in block:
                yield child
                for node in child:
                    yield node
    
    def edges(self):
        for i, block in enumerate(self.children):
            if not block: continue
            for child in block:
                yield (self, i, child)
                for edge in child.edges():
                    yield edge
    
    def rules(self):
        for node in self:
            yield (node.expanded_conditions, node.blocks, round(node.mp, 4), round(node.mr, 4), node.n_instances)




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

'''
for n in get_nodes_by_type(net, (Sum)):
    n.y = np.concatenate([np.zeros(len(child.data)) + j
                          for j, child in enumerate(n.children)])
    n.X = np.concatenate([child.data for child in n.children])
'''
if __name__ == '__main__':
    train, test = get_dataset('adult')


    leaf_type = {
        DataType.CONTINUOUS: Gaussian,
        DataType.CATEGORICAL: Categorical,
        DataType.BINARY: Bernoulli
    }

    ptypes = [leaf_type[t] for t in train.data_types]
    context = Context(parametric_types=ptypes).add_domains(train.instances)

    net = learn_parametric(train.instances,
                        ds_context=context,
                        rows="gmm",
                        min_instances_slice=len(train.instances) / 100)

    for node in get_nodes_by_type(net, (Sum)):
        # print(len(node.X))
        del node.X
        del node.y

    names = train.names
    data_types = train.data_types

    csi = explain(net, train.names, train.data_types, train)


    def print_csi(node):
        print("Node", node, "conditions", node.conditions)
        for child in node.children:
            if child is not None:

                print_csi(child)


    def format_rule(rule, data):
        data2type = dict(zip(data.names, data.data_types))
        antecedent, consequent, *_ = rule
        terms = []
        for term in antecedent:
            subterms = []
            for subterm in term:
                elements = []
                for element in subterm.split(' ^ '):
                    x, symbol, y = element.lstrip('(').rstrip(')').split(' ')
                    if data2type[x] == DataType.BINARY:
                        if (symbol, y) in [('==', '1'), ('!=', '0')]:
                            elements.append(x)
                        else:
                            elements.append('~%s' % x)
                    elif data2type[x] == DataType.CATEGORICAL and data.categories is not None:
                        data2cat = dict(zip(data.names, data.categories))
                        cats = data2cat[x]
                        catname = cats[int(y)]
                        elements.append('%s %s %s' % (x, symbol, catname))
                    else:
                        elements.append('%s %s %s' % (x, symbol, y))
                if len(elements) > 1:
                    elements = ['(%s)' % each for each in elements]
                subterms.append(' ^ '.join(elements))
            if len(subterms) > 1:
                subterms = ['(%s)' % each for each in subterms]
            terms.append(' v '.join(subterms))
        if len(terms) > 1:
            terms = ['(%s)' % each for each in terms]
        antecedent = ' ^ '.join(terms)
        consequent = "[%s]" % ", ".join(["[%s]" % ", ".join(block) for block in consequent])
        return antecedent, consequent
        

    # print_csi(csi)
    # print (csi.to_dot())
    for rule in csi.rules():
        print ("%s => %s" % format_rule(rule, train)) 