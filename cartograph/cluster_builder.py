"""
    Given lists of vectors, outputs a list of article ids, article names, and the countries each article
    belongs to.

    This implementation uses the kmeans ++ algorithm.

    Authors: Jonathan Scott, Lu Li
"""

from sklearn.cluster import KMeans
import pandas as pd
import os


def get_article_vectors(map_directory):
    return pd.read_csv(map_directory + '/article_vectors.csv')


def get_cluster(map_directory):
    article_vectors = get_article_vectors(map_directory)
    article_id = article_vectors['article_id']
    matrix = article_vectors.iloc[:, 2:].as_matrix()
    kmeans = KMeans().fit(matrix[0:len(matrix)])
    article_vectors['country'] = kmeans.labels_
    df = pd.DataFrame(article_vectors, columns=['country'])
    df['article_id'] = article_id
    return df.set_index(df['article_id']).iloc[:, 0:1]


def create_csv(directory):
    df = get_cluster(directory)
    df.to_csv(directory + '/cluster_groups.csv', index_label='article_id')


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    create_csv(map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
