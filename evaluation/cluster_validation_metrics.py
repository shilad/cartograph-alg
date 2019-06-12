"""
Internal evaluation methods on clustering algorithms.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score
import functools
import sklearn.metrics as metrics
from s_dbw import S_Dbw
from hdbscan import HDBSCAN

from scipy import stats



def db_score(xy_list, labels):
    # return davies bouldin score, the smaller the better a cluster is.
    return metrics.cluster.davies_bouldin_score(xy_list, labels)


def silhouette_score(xy_list, labels):
    # return silhouette score, the larger the better a cluster is.
    return metrics.cluster.silhouette_score(xy_list, labels)


def ch_score(xy_list, labels):
    # return silhouette score, the larger the better a cluster is.
    return metrics.cluster.silhouette_score(xy_list, labels)


def read_cluster(map_directory, method):
    original = pd.read_csv(map_directory + '/cluster_groups_' + method + '.csv').sort_values(by=['article_id'])['country']



def main(map_directory, method):
    datapoints = pd.read_csv(map_directory + '/article_vectors_' + method + '.csv').drop(columns='article_name')
    X = datapoints.values
    original_cluster = read_cluster(map_directory, method)
    print(ch_score(X, original_cluster))

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory vector_directory' % sys.argv[0])
        sys.exit(1)

    map_directory, method = sys.argv[1:]
    main(map_directory,  method)




