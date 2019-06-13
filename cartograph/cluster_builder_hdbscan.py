from hdbscan import HDBSCAN
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


def get_article_vectors(map_directory):
    return pd.read_csv(map_directory + '/article_vectors_original.csv')


def get_combined_label_article_vectors(map_directory):
    return pd.read_csv(map_directory + '/article_vectors_original.csv')


def get_cluster(map_directory, method='original'):
    if method == 'original':
        article_vectors = get_article_vectors(map_directory)
        article_id = article_vectors['article_id']
        matrix = article_vectors.iloc[:, 2:].values
        cluster_labels = HDBSCAN(5, min_samples=1, cluster_selection_method="leaf").fit(matrix[0:len(matrix)])
        article_vectors['country'] = cluster_labels.labels_
        df = pd.DataFrame(article_vectors, columns=['country'])
        df['article_id'] = article_id

    if method == 'combined':
        article_vectors = get_combined_label_article_vectors(map_directory)

        article_id = article_vectors['article_id']
        matrix = article_vectors.iloc[:, 2:].values
        cluster_labels = HDBSCAN(25, min_samples=1, cluster_selection_method="leaf").fit_predict(matrix[0:len(matrix)])
        article_vectors['country'] = cluster_labels
        df = pd.DataFrame(article_vectors, columns=['country'])
        df['article_id'] = article_id

    return df.set_index(df['article_id']).iloc[:, 0:1]


def create_csv(directory, method):
    if method == 'original':
        df = get_cluster(directory)
        df.to_csv(directory + '/cluster_groups_original_hdbscan.csv', index_label='article_id')
    if method == 'combined':
        df = get_cluster(directory, method)
        df.to_csv(directory + '/cluster_groups_combined_hdbscan.csv', index_label='article_id')


def main(map_directory, method):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    if method == 'original':
        create_csv(map_directory, method)
    if method == 'combined':
        create_csv(map_directory, method)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory, method = sys.argv[1:]
    main(map_directory, method)
