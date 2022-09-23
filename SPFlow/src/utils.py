import pandas as pd
import numpy as np
import lightgbm
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

def categorical_to_condition(Xi, i, y):
    """Finds a simple decision rule for categorical data"""

    Xi = pd.get_dummies(Xi)
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(Xi, y)
    tree = clf.tree_
    j = tree.feature[0]
    sign = "=" if tree.value[tree.children_right[0]].argmax() == 1 else "!="
    
    return (sign, i, Xi.columns[j])


def continuous_to_condition(Xi, i, y):
    """Finds a simple decision rule for continuous data"""
    Xi = Xi.reshape(-1, 1)
    clf = DecisionTreeClassifier(max_depth=1)
    clf.fit(Xi, y)
    tree = clf.tree_
    t = tree.threshold[0]
    sign = "<=" if tree.value[tree.children_right[0]].argmax() == 0 else ">"
    return (sign, i ,t)


def flip_sign(sign):
    """returns the opposite sign"""
    opposites = { ">": "<=",
                "<=": ">",
                "=": "!=",
                "!=": "=" }
    
    assert sign in opposites.keys()

    return opposites.get(sign)


def to_condition(node, categorical):
    """ Finds a simple conditon to represent a sum node 
        Arguments:
            node: A sum node
            categorical: a boolean list containing whether each variable is categorical (all variables from dataset)
    """
    est = lightgbm.LGBMClassifier(importance_type = "gain")
    cv = GridSearchCV(est, param_grid = {'max_depth': [2, 3, 4], 'n_estimators': list(range(20, 101, 20))})
    
    X, y = node.X, node.y
    cv.fit(X, y, categorical_feature=[i for i, j in enumerate(node.scope) if categorical[j]])
    est = cv.best_estimator_
    i = est.feature_importances_.argmax()
    
    Xi = X[:, i]

    if categorical[node.scope[i]]:
        return categorical_to_condition(Xi, i, y)
    else:
        return continuous_to_condition(Xi, i, y)


if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)

    print ("Test continuous #1")
    X = np.arange(-5, 5, step=0.5)
    y = np.ones_like(X, dtype=int)
    y[X < 1] = 0

    print ("Expected ('>', 1, 0.75)")
    print ("Actual", continuous_to_condition(X, 1, y))

    print ()

    print ("Test continuous #2")
    X = np.arange(-5, 5, step=0.5)
    y = np.ones_like(X, dtype=int)
    y[X > 1] = 0

    print ("Expected ('<=', 1, 1.25)")
    print ("Actual", continuous_to_condition(X, 1, y))

    print ()
    print ("Test categorical #1")
    X = np.random.randint(0, 5, size=20, dtype=int)
    y = np.zeros_like(X)
    y[X == 2] = 1
    print ("Expected ('=', 1, 2)")
    print ("Actual", categorical_to_condition(X, 1, y))