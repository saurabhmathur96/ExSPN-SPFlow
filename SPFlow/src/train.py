from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.leaves.parametric.Parametric import Bernoulli, Categorical, Gaussian
from spn.structure.Base import Context, Sum
from spn.structure.Base import get_nodes_by_type
from dataset import DataType, get_dataset
from argparse import ArgumentParser
from joblib import dump
import numpy as np

np.random.seed(0)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('dataset_name')
    parser.add_argument('save_path')
    parser.add_argument('--precomputed-instance-function', dest='precomputed', action='store_true', default=False)
    args = parser.parse_args()

    leaf_type = {
        DataType.CONTINUOUS: Gaussian,
        DataType.CATEGORICAL: Categorical,
        DataType.BINARY: Bernoulli
    }

    train, test = get_dataset(args.dataset_name)
    
    ptypes = [leaf_type[t] for t in train.data_types]
    context = Context(parametric_types=ptypes).add_domains(train.instances)

    if args.dataset_name == 'artificial':
        min_instances = len(train.instances) / 5
    elif args.dataset_name == 'numom2b':
        min_instances = sum(train.instances[:, -1]) / 100
    else:
        min_instances = len(train.instances) / 100
    net = learn_parametric(train.instances,
                        ds_context=context,
                        rows="gmm",
                        min_instances_slice=min_instances)
    print ("%.2f" % np.mean(log_likelihood(net, test.instances)))

    if not args.precomputed:
        for node in get_nodes_by_type(net, (Sum)):
            # print(len(node.X))
            del node.X
            del node.y
    
    dump(net, args.save_path)