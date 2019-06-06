"""
Given list of vector representations of Wikipedia articles, output a dataframe containing the article id and the vectors.
Author: Lily Irvin, Jonathan Scott
"""

import pandas as pd
import time


def read_domain_concepts(path):
    return pd.read_csv(path+'/domain_concept.csv')


def vec_str_to_float(string):
    try:
        return float(string)
    except ValueError:
        pass


def read_vectors(vec_path):
    vectors = {}
    start = time.time()
    with open(vec_path, encoding="ISO-8859-1") as file:
        for line in file:
            values = line.split()
            vectors[values[0]] = [vec_str_to_float(x) for x in values[1:]]
    end = time.time()
    print("Reading in data takes: "+str(end-start)+" seconds.")
    return vectors


def map_domain_concept_id_to_article_vector(domain_concept_df):
    temp_map_dict = {}  # for fast look up
    for i, row in domain_concept_df.iterrows():
        temp_map_dict[row['article_name'].replace(" ", "_")] = row['article_id']
    #print("mapping of dc to av",len(temp_map_dict))
    return temp_map_dict


def find_intersection_btw_dom_concept_vectors(dom_con_to_art_vecs, vectors):
    article_set = set(dom_con_to_art_vecs.keys())  # O(1) look up.
    article_vectors_df_ready = []
    for article in vectors.keys():
        if article in article_set:
            article_vectors_df_ready.append([article]+vectors[article])
    print("av_df_ready",len(article_vectors_df_ready))
    return article_vectors_df_ready


def create_article_vec_csv(article_vectors_df_ready, domain_concept_df, dom_con_to_art_vecs):
    vector_ids = ['vector_'+str(i) for i in range(100)]  # we know the size of the vectors previously
    #domain_concept_df = pd.read_csv('../data/food/domain_concept.csv')
    for i, row in domain_concept_df.iterrows():
        dom_con_to_art_vecs[row['article_name'].replace(" ", "_")] = row['article_id']

    # a Dataframe expects a dictionary where {col:[list of values in column]
    article_w_vectors = pd.DataFrame(article_vectors_df_ready, columns=['article_name']+vector_ids)
    for i, row in article_w_vectors.iterrows():
        if i == 0:
            article_w_vectors.insert(0, 'article_id', dom_con_to_art_vecs[row['article_name']])
        else:
            article_w_vectors.loc[i, 'article_id'] = dom_con_to_art_vecs[row['article_name']]
    article_w_vectors.sort_values(by=['article_id']).to_csv("../data/article_vectors.csv", index=False) # get article ids from domain_concepts file


def main(map_directory, vector_directory):
    domain_concept_df = read_domain_concepts(map_directory)
    vectors = read_vectors(vector_directory)
    dom_con2vec = map_domain_concept_id_to_article_vector(domain_concept_df)
    article_vectors_df_ready = find_intersection_btw_dom_concept_vectors(dom_con2vec, vectors)
    create_article_vec_csv(article_vectors_df_ready, domain_concept_df, dom_con2vec)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory vector_directory' % sys.argv[0])
        sys.exit(1)

    map_directory, vector_directory = sys.argv[1:]
    main(map_directory, vector_directory)





