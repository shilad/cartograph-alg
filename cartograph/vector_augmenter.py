"""
Given a list of vector representations of Wikipedia articles,
output an augmented dataframe.

If input method is "label", return a combined matrix
that contains article vectors and label svd vectors.

If input method is "cluster", return a combined matrix
that contains article vectors and cluster vectors.

If input method is "all", return a combined matrix
that contains article vectors, cluster vectors, and cluster vectors.

Author: Lily Irvin, Jonathan Scott, Lu, Li
"""

import pandas as pd
import argparse
import numpy as np
from sklearn.decomposition import TruncatedSVD
import sys
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs, LinearOperator


def create_label_matrix(label_matrix):
    """Creates a matrix that contains a article ids and label ids."""
    output_matrix = np.zeros((max(label_matrix['article_id'])+1, max(label_matrix['label_id'])+1))
    for i in range(len(label_matrix['article_id'])):
        current_article = label_matrix.iloc[i].iloc[0]
        output_matrix[current_article][label_matrix.iloc[i][1]] = 1
    output_matrix = pd.DataFrame(output_matrix)
    output_matrix.index.name = 'article_id'
    return output_matrix


def get_label_svd(article_vectors, art_labels):
    label_wide_matrix = csc_matrix(create_label_matrix(art_labels).values, dtype=float)
    lp_mat_reduced, s, vt = svds(label_wide_matrix, k=10)
    reduce_vec_labels = ['svd_'+str(i) for i in range(lp_mat_reduced.shape[1])]
    label_svd = pd.DataFrame({}, columns=['article_id']+reduce_vec_labels)
    for i in range(len(article_vectors)):
        label_svd.loc[i, 'article_id'] = article_vectors.loc[i, 'article_id']
        label_svd.iloc[i, 1:] = lp_mat_reduced[i, :]
    return label_svd


def get_cluster_matrix(cluster_csv):
    num_of_countries = len(cluster_csv['country'].unique())
    cluster_csv = cluster_csv[cluster_csv['country'] != -1]
    cluster_matrix = np.zeros((cluster_csv.shape[0], num_of_countries + 1))
    for i in range(len(cluster_csv['article_id'])):
        cluster_matrix[i][0] = cluster_csv.loc[i, 'article_id']  # assign the first column to be article ids
        country = cluster_csv.loc[i, 'country']
        cluster_matrix[i][country+1] = 1
    country_labels = ['country_' + str(i) for i in range(num_of_countries)]
    cluster_matrix_df = pd.DataFrame(cluster_matrix, columns=['article_id'] + country_labels)
    return cluster_matrix_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Augment the original article vectors with label matrix or '
                                                 'cluster matrix.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--label_vectors', required=True)
    parser.add_argument('--method', required=True)
    parser.add_argument('--cluster_vectors', required=False)
    parser.add_argument('--output_file', required=True)

    args = parser.parse_args()

    article_vectors = pd.read_csv(args.vectors)
    label_csv = pd.read_csv(args.label_vectors)
    label_vectors = get_label_svd(article_vectors, label_csv)

    if args.method == 'label':
        cluster_df = pd.merge(article_vectors, label_vectors, on='article_id')
    elif args.method == 'cluster':
        cluster_csv = pd.read_csv(args.cluster_vectors)
        cluster_vectors = get_cluster_matrix(cluster_csv)
        cluster_df = pd.merge(cluster_vectors, article_vectors, on='article_id')
    elif args.method == 'all':
        cluster_csv = pd.read_csv(args.cluster_vectors)
        cluster_vectors = get_cluster_matrix(cluster_csv)
        cluster_df_with_cluster = pd.merge(cluster_vectors, article_vectors, on='article_id')
        cluster_df = pd.merge(cluster_df_with_cluster, label_vectors, on='article_id')
    else:
        sys.stderr.write("Unkonwn clustering method: %s\n" + args.clustering)
        sys.exit(1)
    cluster_df.to_csv('%s/%s' % (args.experiment, args.output_file), index=False)



