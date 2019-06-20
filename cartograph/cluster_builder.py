"""
Given lists of vectors, outputs a list of article ids and the country that article
belongs to.

This implementation uses the kmeans ++ algorithm.

Author: Lu Li
"""
import sys

from sklearn.cluster import KMeans
import pandas as pd
from hdbscan import HDBSCAN
import argparse


def get_kmeans(article_vectors, k):
    """

    :param article_vectors: vectors to cluster
    :param k: desired number of clusters for kmeans
    :return: a dataframe two columns: article id and country
    """

    article_id = article_vectors['article_id']
    matrix = article_vectors.iloc[:, 2:].values
    kmeans = KMeans(n_clusters=int(k)).fit(matrix[0:len(matrix)])
    article_vectors['country'] = kmeans.labels_

    df = pd.DataFrame(article_vectors, columns=['country'])
    df['article_id'] = article_id
    return df.set_index(df['article_id']).iloc[:, 0:1]


def get_hdbscan(article_vectors, min_size):
    """

    :param article_vectors: vectors to cluster
    :param min_size: minimum sample size for HDBSCAN
    :return: a dataframe two columns: article id and country
    """

    article_id = article_vectors['article_id']
    matrix = article_vectors.iloc[:, 2:].values
    hdbscan = HDBSCAN(min_cluster_size=int(min_size),
                      cluster_selection_method="leaf").fit(matrix[0:len(matrix)])
    article_vectors['country'] = hdbscan.labels_

    df = pd.DataFrame(article_vectors, columns=['country'])
    df['article_id'] = article_id
    return df.set_index(df['article_id']).iloc[:, 0:1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster articles in the high dimensional space using K-means or '
                                                 'hdbscan.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--clustering', required=True)
    parser.add_argument('--k', default=8)
    parser.add_argument('--min_size', default=8)

    args = parser.parse_args()

    article_vectors = pd.read_csv(args.vectors)
    if args.clustering == 'kmeans':
        cluster_df = get_kmeans(article_vectors, args.k)
    elif args.clustering == 'hdbscan':
        cluster_df = get_hdbscan(article_vectors, args.min_size)
    else:
        sys.stderr.write("Unkonwn clustering method: %s\n" + args.clustering)
        sys.exit(1)

    cluster_df.to_csv('%s/cluster_groups.csv' % (args.experiment, ), index_label='article_id')

