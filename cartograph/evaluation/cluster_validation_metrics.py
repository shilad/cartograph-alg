"""
Internal evaluation metrics on validating clustering algorithms.
import sdbw by entering pip install --upgrade s-dbw in terminal.
"""
import argparse
import pandas as pd
import sklearn.metrics as metrics
from pandas._libs import json
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
    return metrics.cluster.calinski_harabasz_score(xy_list, labels)


def main():
    article_vectors = pd.read_csv(args.vectors)
    cluster_groups = pd.read_csv(args.groups).drop(columns=['article_id'])
    if 'country_likelihood' in cluster_groups.columns:
        cluster_groups = pd.read_csv(args.groups).drop(columns=['country_likelihood', 'article_id'])
    silhouette_score = get_silhouette_score(article_vectors, cluster_groups)
    # sdb_w_score = get_sdbw_score(article_vectors, cluster_groups)
    ch_score = get_ch_score(article_vectors, cluster_groups)
    print(str(json.dumps({'silhouette score': silhouette_score})))
    print(str(json.dumps({'ch score:': ch_score})))

    # logging.warning("Modularity Score: %.6f", mod_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment the original article vectors with label matrix or '
                                                 'cluster matrix.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--groups', required=True)
    args = parser.parse_args()
    main()