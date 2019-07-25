"""
Internal evaluation metrics on validating clustering algorithms.
import sdbw by entering pip install --upgrade s-dbw in terminal.
"""
import argparse
import pandas as pd
import sklearn.metrics as metrics
from pandas._libs import json
from s_dbw import S_Dbw
from scipy import spatial
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import adjusted_mutual_info_score
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.utils import check_X_y
from sklearn.preprocessing import LabelEncoder

def get_sdbw_score(xy_list, labels):
    # return S Dbw validity index, the smaller the better a cluster is.
    return S_Dbw(xy_list, labels) # add cosine metric and see Halkidi vs Tong


def get_db_score(xy_list, labels):
    """
    :param xy_list:
    :param labels:
    :return: A value in range [0,inf) The lower the value the better the quality of clustering.
    """
    return metrics.cluster.davies_bouldin_score(xy_list, labels)


def get_silhouette_score(xy_list, labels):
    """
    :param xy_list:
    :param labels:
    :return: A value in range [-1,1] where 1 is the best value,
    -1 is the worst and 0 indicates overlapping clusters.
    """
    return metrics.cluster.silhouette_score(xy_list, labels)


def get_ch_score(xy_list, labels):
    """
    :param xy_list:
    :param labels:
    :return: A float where a higher value indicates better quality of clusters.
    """
    X, labels = check_X_y(xy_list, labels)
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    n_samples, _ = X.shape
    n_labels = len(le.classes_)
    extra_disp, intra_disp = 0., 0.
    mean = np.mean(X, axis=0).reshape(1, -1)
    for k in range(n_labels):
        cluster_k = X[labels == k]
        mean_k = np.mean(cluster_k, axis=0).reshape(1, 101)
        extra_disp += np.sum(len(cluster_k) * cosine_distances(mean_k, mean))
        h = cosine_distances(cluster_k, mean_k)
        intra_disp += np.sum(h)
    return (1. if intra_disp == 0. else
            extra_disp * (n_samples - n_labels) /
            (intra_disp * (n_labels - 1.)))


def main(experiment, vectors, clusterA, clusterB=None):
    article_vectors = pd.read_csv(vectors)
    cluster_a = pd.read_csv(clusterA).drop(columns=['article_id'])
    if 'distance' in cluster_a.columns:
        cluster_a = pd.read_csv(clusterA).drop(columns=['distance', 'article_id'])

    if not clusterB:
        ch_score = get_ch_score(article_vectors,
                                cluster_a.values.ravel())
        silhouette_score = get_silhouette_score(article_vectors,
                                                cluster_a.values.ravel())
        print(str(json.dumps({'ch score:': ch_score})))
        print(str(json.dumps({'silhouette_score': silhouette_score})))

    elif clusterB:
        cluster_a = pd.read_csv(experiment + '/original_cluster_groups.csv')
        cluster_b = pd.read_csv(clusterB)
        rand_index = adjusted_rand_score(cluster_a['country'], cluster_b['country'])
        mutual_info = adjusted_mutual_info_score(cluster_a['country'], cluster_b['country'])
        print(str(json.dumps({'rand_index score': rand_index})))
        print(str(json.dumps({'mutal_info': mutual_info})))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Measure a cluster\'s '
                                                 'quality or compare clusterings.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--cluster_A', required=True)
    parser.add_argument('--cluster_B', required=False)
    args = parser.parse_args()
    main(args.experiment, args.vectors, args.cluster_A, args.cluster_B)
