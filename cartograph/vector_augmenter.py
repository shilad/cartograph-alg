"""
Given a list of vector representations of Wikipedia articles,
output a data frame containing the article id and the vectors.

If input method is "combined", return a combined matrix
that contains article vectors as well as label vectors.

Author: Lily Irvin, Jonathan Scott, Lu, Li
"""

import pandas as pd
import time
import numpy as np
from functools import reduce
from sklearn.decomposition import TruncatedSVD


def read_vectors(path):
    return pd.read_csv(path + '/article_vectors_original.csv')


def create_article_vec_csv(article_vectors, map_directory, method = "no_svd"):

    if method == "svd_pop":
        svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        art_labels = pd.read_csv(map_directory + '/article_categories.csv')
        label_wide_matrix = create_label_matrix(art_labels)
        pop_matrix = pd.read_csv(map_directory + '/popularity_score.csv')

        label_pop_matrix = pd.merge(label_wide_matrix, pop_matrix, on='article_id')

        lp_mat_reduced = svd.fit_transform(X=label_pop_matrix.iloc[:, 1:].to_numpy())
        reduce_vec_labels = ['svd_'+str(i) for i in range(lp_mat_reduced.shape[1])]
        combined_matrix = pd.DataFrame({}, columns=['article_id']+reduce_vec_labels)
        #print(article_vectors.shape,len(article_vectors))
        for i in range(len(article_vectors)):
            combined_matrix.loc[i, 'article_id'] = article_vectors.loc[i, 'article_id']
            combined_matrix.iloc[i, 1:] = lp_mat_reduced[i, :]

        #df_list = [article_vectors, combined_matrix, pop_matrix]
        #combined_matrix = reduce(lambda df1, df2: pd.merge(df1, df2, on='article_id'), df_list)
        combined_matrix = pd.merge(article_vectors, combined_matrix, on='article_id')
        combined_matrix.to_csv(map_directory + "/article_vectors_" + method + ".csv", index=False)

    elif method == 'no_svd':
        art_labels = pd.read_csv(map_directory + '/article_categories.csv')
        label_wide_matrix = create_label_matrix(art_labels)
        #pop_matrix = pd.read_csv(map_directory + '/popularity_score.csv')

        df_list = [article_vectors, label_wide_matrix]
        combined_matrix = reduce(lambda df1, df2: pd.merge(df1, df2, on='article_id'), df_list)
        combined_matrix.to_csv(map_directory + "/article_vectors_" + method + ".csv", index=False)
    print('Finished Augmenting Vectors!')
    return


def create_label_matrix(label_matrix):
    """Creates a matrix that contains a article ids and label ids."""
    output_matrix = np.zeros((max(label_matrix['article_id'])+1, max(label_matrix['label_id'])+1))
    for i in range(len(label_matrix['article_id'])):
        current_article = label_matrix.iloc[i].iloc[0]
        output_matrix[current_article][label_matrix.iloc[i][1]] = 1
    output_matrix = pd.DataFrame(output_matrix)
    output_matrix.index.name = 'article_id'
    return output_matrix


def main(map_directory, method):
    vectors = read_vectors(map_directory)
    create_article_vec_csv(vectors, map_directory, method)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory augmentation_method' % sys.argv[0])
        sys.exit(1)

    map_directory, method = sys.argv[1:]
    main(map_directory, method)





