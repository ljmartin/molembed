import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, cdist, squareform
import copy

def getSeed(seed=25038):
    """Sets a so that splits and random selections are the same across iterations."""
    return seed


def load_feature_and_label_matrices(type='morgan'):
    """Loads the featurized chemicals (x) and the label matrix (y)
    Returns them as sparse matrices because they're mostly zero."""
    y = sparse.load_npz('../data/interaction_matrix_pchembl.npz').toarray()
    x = sparse.load_npz('../data/'+type+'.npz')
    return x, y


def get_subset(x, y, indices):
    """For methods development we may not need the full dataset. It's faster
    to use a subset. After choosing some random column indices, this parses
    `x` and `y` to keep ligands associated with those targets and remove all
    else"""
    y_ = y[:,indices]
    #remove ligands that do not have a positive label in the subset
    row_mask = y_.sum(axis=1)>0
    y_ = y_[row_mask]
    x_ = x[row_mask]
    return x_, y_


def fast_dice(X, Y=None):
    """Dice distances between binary-valued features can be achieved much faster
    using sparse matrices. This calculates sparse dice distance matrices between
    all ligands in either `X`, or between `X` and `Y`. It's like cdist."""
    if isinstance(X, np.ndarray):
        X = sparse.csr_matrix(X).astype(bool).astype(int)
    if Y is None:
        Y = X
    else:
        if isinstance(Y, np.ndarray):
            Y = sparse.csr_matrix(Y).astype(bool).astype(int)
            
    intersect = X.dot(Y.T)
    #cardinality = X.sum(1).A
    cardinality_X = X.getnnz(1)[:,None] #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    cardinality_Y = Y.getnnz(1) #slightly faster on large matrices - 13s vs 16s for 12k x 12k
    return (1-(2*intersect) / (cardinality_X+cardinality_Y.T)).A
