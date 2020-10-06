#normal tools:
import numpy as np
import copy
import sys
sys.path.append("..")
import utils
import numpy as np
from scipy import stats, sparse
from tqdm import tqdm

from mccv import MCCVSplitter




##Set a random seed to make it reproducible!
np.random.seed(utils.getSeed())
#load up data:
x, y = utils.load_feature_and_label_matrices(type='morgan')
##select a subset of columns of 'y' to use as a test matrix:
#this is the same each time thanks to setting the random.seed.
col_indices = np.random.choice(y.shape[1], 100, replace=False)
x_, y_ = utils.get_subset(x, y, indices=col_indices)


#load the pairwise distance matrix:
ecfp_distance_matrix = np.memmap('./morgan_distance_matrix.dat', dtype=np.float16, mode='r', 
                                 shape=(x_.shape[0], x_.shape[0]))
#load the adjacency graph (used for clustering)
adjacency = sparse.load_npz('./knn_graph.npz')


splitter = MCCVSplitter(x_, y_, adjacency, ecfp_distance_matrix)
splitter.cluster()
