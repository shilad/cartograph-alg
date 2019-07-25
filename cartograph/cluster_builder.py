"""
Given lists of vectors, outputs a list of article ids and the country that article
belongs to.

This implementation uses the kmeans ++ algorithm.

Author: Lu Li
"""
import sys
import math
from scipy import spatial
from sklearn.cluster import KMeans
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import argparse
import numpy as np
from hdbscan import HDBSCAN


def mag(x):
    return math.sqrt(sum(i**2 for i in x))


def normalize_vectors(dataframe):
    mags = []
    for index, row in dataframe.iterrows():
        vector = row[1:100]
        mag1 = mag(vector)
        mags.append(mag1)
    dataframe['magnitude'] = mags

    try:
         dataframe.iloc[:, 1:100] = dataframe.iloc[:, 1:100].div(dataframe['magnitude'].values, axis=0) * 1000
    except ZeroDivisionError:
        print("divide by zero")

    return dataframe


def get_kmeans(article_vectors, k):
    """

    :param article_vectors: vectors to cluster
    :param k: desired number of clusters for kmeans
    :return: a dataframe two columns: article id and country
    """
    article_id = article_vectors['article_id']
    # creates a 4004 x 100 matrix
    matrix = article_vectors.iloc[:, 1:].values
    kmeans = KMeans(n_clusters=int(k)).fit(matrix[0:len(matrix)])
    article_vectors['country'] = kmeans.labels_
    # creates a 4004 x 2 dataframe with article id and the country it belongs to.
    df = pd.DataFrame(article_vectors, columns=['country'])
    df['article_id'] = article_id
    centroids = pd.DataFrame(list(kmeans.cluster_centers_))
    centroids.index.name = 'country'
    df_with_vectors = pd.merge(df, article_vectors.drop(columns='country'), on='article_id')
    df_with_centroids = pd.merge(df_with_vectors, centroids, on='country')
    dist = []
    for index, row in df_with_centroids.iterrows():
        num_dims = (len(row) - 2) / 2
        a = np.array(row[2: int(len(row) - num_dims)])
        b = np.array(row[int(len(row) - num_dims): len(row)])
        dist.append(spatial.distance.cosine(a, b))
        # dist.append(np.linalg.norm(a - b))
    df_with_centroids['distance'] = dist
    df_with_centroids = df_with_centroids[['country', 'article_id', 'distance']]
    df_with_centroids = df_with_centroids.sort_values(by=['article_id'])
    return df_with_centroids.reset_index(drop=True)


def get_hdbscan(article_vectors, min_size, noise=False):
    """

    :param article_vectors: vectors to cluster
    :param min_size: minimum sample size for HDBSCAN
    :return: a dataframe two columns: article id and country
    """

    article_id = article_vectors['article_id']
    matrix = normalize_vectors(article_vectors.iloc[:, 2:]).values
    hdbscan = HDBSCAN(min_cluster_size=int(min_size), alpha=1.3, cluster_selection_method='leaf', min_samples=8).fit(matrix[0:len(matrix)])
    article_vectors['country'] = hdbscan.labels_
    df = pd.DataFrame(article_vectors, columns=['country'])
    df['article_id'] = article_id
    if noise:
        probabilities = hdbscan.probabilities_
        df['country_likelihood'] = probabilities
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster articles in the high dimensional space using K-means or '
                                                 'hdbscan.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--clustering', required=True)
    parser.add_argument('--k', default=8)
    parser.add_argument('--min_size', default=8)

    args = parser.parse_args()
    # article_vectors = normalize_vectors(pd.read_csv(args.vectors))
    article_vectors = pd.read_csv(args.vectors)
    if args.clustering == 'kmeans':
        cluster_df = get_kmeans(article_vectors, args.k)
    elif args.clustering == 'hdbscan':
        cluster_df = get_hdbscan(article_vectors, args.min_size)
    elif args.clustering == 'noise':
        kmeans = get_kmeans(article_vectors, args.k)
        hdbscan = get_hdbscan(article_vectors, args.min_size, True)
        hdbscan = hdbscan.drop(columns=['country'])
        cluster_df = pd.merge(hdbscan, kmeans, on='article_id')
    else:
        sys.stderr.write("Unkonwn clustering method: %s\n" + args.clustering)
        sys.exit(1)

    cluster_df.to_csv('%s/cluster_groups.csv' % (args.experiment, ), index=False)

