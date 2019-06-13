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


def read_domain_concepts(path):
    return pd.read_csv(path+'/domain_concept.csv')


def vec_str_to_float(string):
    try:
        return float(string)
    except ValueError:
        pass


def read_vectors(vec_path):
    vectors = {}
    #start = time.time()
    with open(vec_path, encoding="ISO-8859-1") as file:
        for line in file:
            values = line.split()
            vectors[values[0]] = [vec_str_to_float(x) for x in values[1:]]
    #end = time.time()
    # print("Reading in data takes: "+str(end-start)+" seconds.")
    return vectors


def map_domain_concept_id_to_article_vector(domain_concept_df):
    temp_map_dict = {}  # for fast look up
    for i, row in domain_concept_df.iterrows():
        temp_map_dict[row['article_name'].replace(" ", "_")] = row['article_id']
    return temp_map_dict


def find_intersection_btw_dom_concept_vectors(dom_con_to_art_vecs, vectors):
    article_set = set(dom_con_to_art_vecs.keys())  # O(1) look up.
    article_vectors_df_ready = []
    for article in vectors.keys():
        if article in article_set:
            article_vectors_df_ready.append([article]+vectors[article])
    return article_vectors_df_ready


def create_article_vec_csv(article_vectors_df_ready, domain_concept_df, dom_con_to_art_vecs, map_directory, method = "original"):
    """
    If method is "original", returns a matrix containing article ids and their vectors;
    If method is "combined", return a matrix with article ids, vectors, and label ids.
    """
    vector_ids = ['vector_'+str(i) for i in range(100)]  # we know the size of the vectors previously
    for i, row in domain_concept_df.iterrows():
        # assigning article_id from domain_concept.csv
        dom_con_to_art_vecs[row['article_name'].replace(" ", "_")] = row['article_id']

    # a Dataframe expects a dictionary where {col:[list of values in column]
    article_w_vectors = pd.DataFrame(article_vectors_df_ready, columns=['article_name']+vector_ids)
    for i, row in article_w_vectors.iterrows():
        if i == 0:
            article_w_vectors.insert(0, 'article_id', dom_con_to_art_vecs[row['article_name']])
        else:
            article_w_vectors.loc[i, 'article_id'] = dom_con_to_art_vecs[row['article_name']]
    article_w_vectors = article_w_vectors.drop(columns='article_name')
    article_w_vectors.sort_values(by=['article_id']).to_csv(map_directory + "/article_vectors_original.csv", index=False)
    # need to specify method in the shell script
    if method == "combined":
        svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
        art_labels = pd.read_csv(map_directory + '/article_labels.csv')
        label_wide_matrix = create_label_matrix(art_labels)
        pop_matrix = pd.read_csv(map_directory + '/popularity_score.csv')

        label_pop_matrix = pd.merge(label_wide_matrix, pop_matrix, on='article_id')
        lp_mat_reduced = svd.fit_transform(X=label_pop_matrix[:, 1:].values())



        df_list = [art_labels, label_wide_matrix, pop_matrix]
        combined_matrix = reduce(lambda df1, df2: pd.merge(df1, df2, on='article_id'), df_list)
        #combined_matrix = pd.merge(article_w_vectors, label_wide_matrix, on='article_id')
        combined_matrix.to_csv(map_directory + "/article_vectors_combined.csv", index=False)

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


def main(map_directory, vector_directory, method):
    domain_concept_df = read_domain_concepts(map_directory)
    vectors = read_vectors(vector_directory)
    dom_con2vec = map_domain_concept_id_to_article_vector(domain_concept_df)
    article_vectors_df_ready = find_intersection_btw_dom_concept_vectors(dom_con2vec, vectors)
    create_article_vec_csv(article_vectors_df_ready, domain_concept_df, dom_con2vec, map_directory, method)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s map_directory vector_directory' % sys.argv[0])
        sys.exit(1)

    map_directory, vector_directory, method = sys.argv[1:]
    main(map_directory, vector_directory, method)





