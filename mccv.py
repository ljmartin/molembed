import numpy as np
from scipy import sparse
from sknetwork.hierarchy import Paris

def get_labels(dendrogram: np.ndarray, cluster: dict, sort_clusters: bool, return_dendrogram: bool):
    """Returns the cluster labels from cluster dict as an array"""
    n = dendrogram.shape[0] + 1
    n_clusters = len(cluster)
    clusters = list(cluster.values())
    index = None
    if sort_clusters:
        sizes = np.array([len(nodes) for nodes in clusters])
        index = np.argsort(-sizes)
        clusters = [clusters[i] for i in index]
    labels = np.zeros(n, dtype=int)
    for label, nodes in enumerate(clusters):
        labels[nodes] = label
    return labels
    
class MCCVSplitter(object):
    def __init__(self, x, y, adj, dmat):
        self.x = x
        self.y = y
        self.adj = adj
        self.dmat = dmat

    def cluster(self):
        self.paris = Paris()
        self.paris.fit(self.adj)
        self.dendrogram = self.paris.dendrogram_

        self.clustered = True

    def cut_balanced(self, max_cluster_size):    
        n = self.dendrogram.shape[0] + 1
        cluster = {i: [i] for i in range(n)}
        for t in range(n - 1):
            i = int(self.dendrogram[t][0])
            j = int(self.dendrogram[t][1])
            if i in cluster and j in cluster and len(cluster[i]) + len(cluster[j]) <= max_cluster_size:
                cluster[n + t] = cluster.pop(i) + cluster.pop(j)        
        self.clusters = get_labels(self.dendrogram, cluster, True, False)


    def split_clusters(self, pos_labels, neg_labels, pos_test_fraction, neg_test_fraction, shuffle=True):
        if shuffle:
            #Shuffle so we can do random selection of clusters:
            np.random.shuffle(pos_labels)
            np.random.shuffle(neg_labels)
        #count number of clusters:
        num_pos_clusters = len(pos_labels)
        num_neg_clusters = len(neg_labels)

        #get test and train positives:
        test_pos_clusters = pos_labels[:max(1,round(num_pos_clusters*pos_test_fraction))]
        train_pos_clusters = pos_labels[max(1,round(num_pos_clusters*pos_test_fraction)):]

        if isinstance(neg_test_fraction, float):
            #get test and train negatives:
	    test_neg_clusters = neg_labels[:int(num_neg_clusters*neg_test_fraction)]
            train_neg_clusters = neg_labels[int(num_neg_clusters*neg_test_fraction):]
        else:
            if sum(neg_test_fraction)>1:
                raise ValueError('Sum of test proportion and train proportion must be less than 1')
            test_neg_clusters = neg_labels[:round(num_neg_clusters*neg_test_fraction[0])]
            train_neg_clusters = neg_labels[-round(num_neg_clusters*neg_test_fraction[1]):]

            #combined:
        test_clusters = list(test_pos_clusters)+list(test_neg_clusters)
        train_clusters = list(train_pos_clusters)+list(train_neg_clusters)

        return test_clusters, train_clusters

        
    def split_random_target(self):
        target_idx = np.random.choice(self.y.shape[1])
        
        all_unique_labels = np.unique(self.clusters)
        pos_labels = np.unique(self.clusters[self.y[:,target_idx]==1])
        neg_labels = all_unique_labels[~np.isin(all_unique_labels, pos_labels)]

        if min(len(pos_labels), len(neg_labels))<2:
            raise SplitError('this target doesnt have enough clusters to split. Somethings gone wrong because this shouldnt happen.')
        
        test_clusters, train_clusters = self.split_clusters(pos_labels, neg_labels, 0.2, [0.1,0.3], shuffle=True)
        return target_idx, test_clusters, train_clusters
            


    def calc_AVE_quick(self, dmat, actives_train, actives_test, inactives_train, inactives_test, decomposed=False):
        inactive_dmat = dmat[inactives_test]
        iTest_iTrain_D = inactive_dmat[:,inactives_train].min(1)
        iTest_aTrain_D = inactive_dmat[:,actives_train].min(1)
        
        active_dmat = dmat[actives_test]
        aTest_aTrain_D = active_dmat[:,actives_train].min(1)
        aTest_iTrain_D = active_dmat[:,inactives_train].min(1)

        aTest_aTrain_S = np.mean( [ np.mean( aTest_aTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )
        aTest_iTrain_S = np.mean( [ np.mean( aTest_iTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )
        iTest_iTrain_S = np.mean( [ np.mean( iTest_iTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )
        iTest_aTrain_S = np.mean( [ np.mean( iTest_aTrain_D < t ) for t in np.linspace( 0, 1.0, 50 ) ] )

        if decomposed:
            return aTest_aTrain_S, aTest_iTrain_S, iTest_iTrain_S, iTest_aTrain_S
        else:
            ave = aTest_aTrain_S-aTest_iTrain_S+iTest_iTrain_S-iTest_aTrain_S
        return ave
