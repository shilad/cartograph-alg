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



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster articles in the high dimensional space using K-means or '
                                                 'hdbscan.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--k', default=8)
    parser.add_argument('--output_file', default=8)

    args = parser.parse_args()
    # article_vectors = normalize_vectors(pd.read_csv(args.vectors))

    article_vectors = pd.read_csv(args.vectors)
    print('max article id is ', article_vectors['article_id'].max())

    cluster_df = get_kmeans(article_vectors, args.k)

    print('max article id is ', cluster_df['article_id'].max())

    cluster_df.to_csv(args.output_file, index=False)

