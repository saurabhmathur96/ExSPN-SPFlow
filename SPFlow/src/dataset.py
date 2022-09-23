from sklearn.datasets import fetch_openml
from collections import namedtuple
import enum
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

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
        data = np.stack(cols, axis=1)
        train, test = train_test_split(data, stratify = frame['class'], random_state = 0)
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

        train, test = train_test_split(D, random_state = 0)

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
            instances=test,
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
            instances=test,
            categories=None
        )
        return train_set, test_set
    elif name == "asia":
        frame = pd.read_csv("asia_data.csv")
        data = frame.to_numpy()
        names = frame.columns.tolist()

        data_types = [DataType.BINARY for n in names]

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
    elif name == "cancer":
        frame = pd.read_csv("cancer_data.csv")
        data = frame.to_numpy()
        names = frame.columns.tolist()

        data_types = [DataType.BINARY for n in names]

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
    elif name == "earthquake":
        frame = pd.read_csv("earthquake_data.csv")
        data = frame.to_numpy()
        names = frame.columns.tolist()

        data_types = [DataType.BINARY for n in names]

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

    else:
        raise ValueError('invalid name')
