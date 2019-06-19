"""
Given lists of vectors, outputs a list of article ids and the country that article
belongs to.

This implementation uses the kmeans ++ algorithm.

Author: Lu Li
"""

from sklearn.cluster import KMeans
import pandas as pd
import os
from hdbscan import HDBSCAN


def get_cluster(article_vectors, cluster_algorithm, k):
    """

    :param article_vectors: vectors to cluster
    :param cluster_algorithm: kmeans or hdbscan
    :param k: desired number of clusters for kmeans or minimum sample size for HDBSCAN
    :return: a dataframe two columns: article id and country
    """

    article_id = article_vectors['article_id']
    matrix = article_vectors.iloc[:, 2:].values
    if cluster_algorithm == 'kmeans':
        kmeans = KMeans(n_clusters=int(k)).fit(matrix[0:len(matrix)])
        article_vectors['country'] = kmeans.labels_
    if cluster_algorithm == 'hdbscan':
        hdbscan = HDBSCAN(min_cluster_size=int(k), cluster_selection_method="leaf").fit(matrix[0:len(matrix)])
        article_vectors['country'] = hdbscan.labels_

    df = pd.DataFrame(article_vectors, columns=['country'])
    df['article_id'] = article_id
    return df.set_index(df['article_id']).iloc[:, 0:1]


def main(directory, vec_method, cluster_algorithm, k):
    """

    :param directory: directory of data files
    :param vec_method: either original or svd
    :param cluster_algorithm: kmeans or hdbscan
    :return: a csv file with two columns: article id and country
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    article_vectors = pd.read_csv('%s/article_vectors_%s.csv' % (directory, vec_method))
    cluster_df = get_cluster(article_vectors, cluster_algorithm, k)
    cluster_df.to_csv('%s/cluster_groups_%s_%s.csv' % (directory, vec_method, cluster_algorithm),
                      index_label='article_id')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory, method, algorithm, k = sys.argv[1:]
    main(map_directory, method, algorithm, k)
