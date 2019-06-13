"""
Internal evaluation metrics on validating clustering algorithms.
"""
import pandas as pd
import sklearn.metrics as metrics
from s_dbw import S_Dbw


def get_sdbw_score(xy_list, labels):
    # return S Dbw validity index, the smaller the better a cluster is.
    return S_Dbw(xy_list, labels) # add cosine metric and see Halkidi vs Tong

def get_db_score(xy_list, labels):
    """
    :param xy_list:
    :param labels:
    :return: A value in range [0,inf) The lower the value the better the quality of clustering.
    """
    # return davies bouldin score, the smaller the better a cluster is.
    return metrics.cluster.davies_bouldin_score(xy_list, labels)


def get_silhouette_score(xy_list, labels):
    """
    :param xy_list:
    :param labels:
    :return: A value in range [-1,1] where 1 is the best value,
    -1 is the worst and 0 indicates overlapping clusters.
    """
    # return silhouette score, the larger the better a cluster is.
    return metrics.cluster.silhouette_score(xy_list, labels)


def get_ch_score(xy_list, labels):
    """
    :param xy_list:
    :param labels:
    :return: A float where a higher value indicates better quality of clusters.
    """
    # return silhgit aouette score, the larger the better a cluster is.
    return metrics.cluster.calinski_harabasz_score(xy_list, labels)


def read_cluster(map_directory, method):
    original = pd.read_csv(map_directory + '/cluster_groups_' + method + '.csv').sort_values(by=['article_id'])['country']
    return original


def get_data_points(map_directory, method):
    datapoints = pd.read_csv(map_directory + '/article_vectors_' + method + '.csv')
    return datapoints.values

def compare_across_metrics(map_directory, algorithm_1, algorithm_2):
    X_1 = get_data_points(map_directory, algorithm_1)
    cluster_1 = read_cluster(map_directory, algorithm_1)
    sdbw_score_1 = get_sdbw_score(X_1, cluster_1)
    silhouette_score_1 = get_silhouette_score(X_1, cluster_1)
    ch_score_1 = get_ch_score(X_1, cluster_1)
    db_score_1 = get_db_score(X_1, cluster_1)

    X_2 = get_data_points(map_directory, algorithm_2)
    cluster_2 = read_cluster(map_directory, algorithm_2)
    sdbw_score_2 = get_sdbw_score(X_2, cluster_2)
    silhouette_score_2 = get_silhouette_score(X_2, cluster_2)
    ch_score_2 = get_ch_score(X_2, cluster_2)
    db_score_2 = get_db_score(X_2, cluster_2)

    if sdbw_score_1 > sdbw_score_2:
        print(algorithm_2 + " gives better clustering according to s_DBw index. ")
    else:
        print(algorithm_1 + " gives better clustering according to s_DBw index. ")

    if silhouette_score_1 > silhouette_score_2:
        print(algorithm_1 + " gives better clustering according to silhouette index. ")
    else:
        print(algorithm_2 + " gives better clustering according to silhouette index. ")

    if db_score_1 > db_score_2:
        print(algorithm_2 + " gives better clustering according to Davies-Bouldin index. ")
    else:
        print(algorithm_1 + " gives better clustering according to Davies-Bouldin index. ")

    if ch_score_1 > ch_score_2:
        print(algorithm_1 + " gives better clustering according to Calinski Harabasz index. ")
    else:
        print(algorithm_2 + " gives better clustering according to Calinski Harabasz index. ")

    return




map_directory = '../data/food'
method = 'original'

compare_across_metrics(map_directory, "original", "combined")