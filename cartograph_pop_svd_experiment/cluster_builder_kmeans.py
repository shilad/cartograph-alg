"""
Given lists of vectors, outputs a list of article ids and the country each article
belongs to.

This implementation uses the kmeans ++ algorithm.

Author: Lu Li, Jonathan Scott
"""

from sklearn.cluster import KMeans
import pandas as pd
import os


def get_article_vectors(map_directory, augmentation):
    return pd.read_csv(map_directory + '/article_vectors_'+augmentation+'.csv')


def get_cluster(article_vectors):
        matrix = article_vectors.iloc[:, 1:].to_numpy()
        kmeans = KMeans().fit(matrix)
        cluster_df = pd.DataFrame({'article_id': article_vectors['article_id'].to_numpy(),
                                   'country': kmeans.labels_})

        return cluster_df


def create_csv(map_directory, method):
        article_vectors = get_article_vectors(map_directory, method)
        article_vectors_clustered = get_cluster(article_vectors)
        article_vectors_clustered.to_csv(map_directory + '/cluster_groups_'+method+'.csv',index=False)


def main(map_directory, method):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    create_csv(map_directory, method)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory augmentation_method' % sys.argv[0])
        sys.exit(1)

    map_directory, method = sys.argv[1:]
    main(map_directory, method)
