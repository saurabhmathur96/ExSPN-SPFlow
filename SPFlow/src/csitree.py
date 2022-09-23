from dataset import DataType

class CSITree:
    def __init__(self, name, conditions, expanded_conditions, blocks, children, mp, mr, n_instances, condition_valid):
        self.name = name
        self.expanded_conditions = expanded_conditions
        self.conditions = conditions
        self.blocks = blocks
        self.children = children
        self.mp = mp
        self.mr = mr
        self.n_instances = n_instances
        self.condition_valid = condition_valid

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
                partial = ' ∨ '.join(term)
                if len(child.conditions) > 1:
                    partial = "(%s)" % partial
                lterms.append(partial)
            label = ' ∧ '.join(lterms)
            if not child.condition_valid:
                label = 'NA'
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
        if len(self.blocks) > 1:
            yield (self.expanded_conditions, self.blocks, round(self.mp, 4), round(self.mr, 4), self.n_instances, self.condition_valid)
        for node in self:
            yield (node.expanded_conditions, node.blocks, round(node.mp, 4), round(node.mr, 4), node.n_instances, node.condition_valid)




def print_csi(node):
    print("Node", node, "conditions", node.conditions)
    for child in node.children:
        if child is not None:

            print_csi(child)


def format_rule(rule, data):
    data2type = dict(zip(data.names, data.data_types))
    antecedent, consequent, *_, condition_is_valid = rule
    if not condition_is_valid:
        consequent = "[%s]" % ", ".join(["[%s]" % ", ".join(block) for block in consequent])
        return 'N/A', consequent
    terms = []
    for term in antecedent:
        subterms = []
        for subterm in term:
            elements = []
            for element in subterm.split(' ^ '):
                if element == '':
                    # elements.append('(N/A)')
                    continue
                x, symbol, y = element.lstrip('(').rstrip(')').split(' ')
                if data2type[x] == DataType.BINARY:
                    if (symbol, y) in [('==', '1'), ('!=', '0')]:
                        elements.append(x)
                    else:
                        elements.append('¬%s' % x)
                elif data2type[x] == DataType.CATEGORICAL and data.categories is not None:
                    data2cat = dict(zip(data.names, data.categories))
                    cats = data2cat[x]
                    # print (x, symbol, y, cats)
                    catname = cats[int(y)]
                    elements.append('%s %s %s' % (x, symbol, catname))
                else:
                    elements.append('%s %s %s' % (x, symbol, y))
            if len(elements) > 1:
                elements = ['(%s)' % each for each in elements]
            subterms.append(' ∧ '.join(elements))
        if len(subterms) > 1:
            subterms = ['(%s)' % each for each in subterms]
        terms.append(' ∨ '.join(subterms))
    if len(terms) > 1:
        terms = ['(%s)' % each for each in terms]
    antecedent = ' ∧ '.join(terms)
    consequent = "[%s]" % " | ".join(["%s" % ",".join(block) for block in consequent])
    return antecedent, consequent