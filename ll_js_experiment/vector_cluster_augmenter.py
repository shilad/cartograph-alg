"""

Given article vectors and their corresponding countries,
outputs a file containing original vectors and appended country adjacency matrix.

Author: Lu Li

"""

import pandas as pd
import numpy as np
import os


def create_cluster_art_csv(article_vec_csv, cluster_csv):
    num_of_countries = len(cluster_csv['country'].unique())
    cluster_matrix = np.zeros((cluster_csv.shape[0], num_of_countries + 1))
    for i in range(len(cluster_csv['article_id'])):
        cluster_matrix[i][0] = cluster_csv.iloc[i].iloc[0]  # assign the first column to be article ids
        offset = 1 - min(cluster_csv['country'].unique())  # HDBSCAN cluster labels start from -1 instead of 0
        cluster_matrix[i][cluster_csv.iloc[i][1] + offset] = 1
    country_labels = ['country_' + str(i) for i in range(num_of_countries)]
    cluster_df = pd.DataFrame(cluster_matrix, columns=['article_id'] + country_labels)
    return pd.merge(article_vec_csv, cluster_df, on='article_id')


def main(directory, vec_method, cluster_algorithm):
    """

    :param directory: directory of file
    :param vec_method: either "original" or "svd" for the one augmented with label svd.
    :param cluster_algorithm: either "kmeans" or "hdbscan"
    :return:
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    art_vec_csv = pd.read_csv("%s/article_vectors_original.csv" % directory)
    cluster_csv = pd.read_csv("%s/cluster_groups_%s_%s.csv" % (directory, vec_method, cluster_algorithm))
    create_cluster_art_csv(art_vec_csv, cluster_csv).to_csv("%s/article_vectors_cluster_%s_%s.csv" % (directory, vec_method, cluster_algorithm), index=False)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s map_directory augmentation_method' % sys.argv[0])
        sys.exit(1)
    map_directory, method, algorithm = sys.argv[1:]
    main(map_directory, method, algorithm)


