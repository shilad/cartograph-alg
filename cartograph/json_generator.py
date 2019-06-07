"""
This script merges the domain concepts, popularity scores, article vectors,
article labels, countries (clusters) and x,y embeddings created in similarly named files
to generate a JSON output where each key is an article/domain concept id and the value
is a dictionary containing each of the described attributes.
Author: Jonathan Scott
"""

import json
import pandas as pd
from functools import reduce


def get_articles(map_directory):
    return pd.read_csv(map_directory+'domain_concept.csv')


def get_popularity_scores(map_directory):
    return pd.read_csv(map_directory + 'popularity_score.csv')


def get_vectors(map_directory):
    return pd.read_csv(map_directory + 'article_vectors.csv')


def get_labels(map_directory):
    return pd.read_csv(map_directory + 'article_labels.csv')


def get_countries(map_directory):
    return pd.read_csv(map_directory + 'cluster_groups.csv')


def get_xy(map_directory):
    return pd.read_csv(map_directory + 'xy_embeddings.csv')


def list_results(map_directory):
    domain_concepts_df = get_articles(map_directory)
    pop_scores_df = get_popularity_scores(map_directory)
    vectors_df = get_vectors(map_directory)
    countries_df = get_countries(map_directory)
    xy_df = get_xy(map_directory)

    return [domain_concepts_df, pop_scores_df, vectors_df,
            countries_df, xy_df]


def merge_results(df_list):
    df = reduce(lambda df1,df2: pd.merge(df1,df2,on='article_id'), df_list) #reduce(pd.merge, df_list)
    return df


def get_vec_as_list(idx,df):
    vec_list = []
    for i in range(100):
        vec_list.append(df.loc[idx, 'vector_' + str(i)])
    return vec_list


def get_labels_as_list(id, label_df):
    """
    This function serves the purpose to reformatting the label_id.csv into
    a form that facillitates JSON generation.
    :param id:
    :param label_df:
    :return:
    """
    label_list = []
    filtered_labels = label_df['label_id'][label_df['article_id'] == id]
    for i in filtered_labels :
        label_list.append(i)
    return label_list


def create_list_article_data(merged_df, map_directory):
    #print(merged_df.columns)
    labels_df = get_labels(map_directory)
    article_data = {}
    for i, row in merged_df.iterrows():
        article_data[row['article_id']] = {'Article': row['article_name_x'], # Should clean this up maybe get rid of one of the columns my fault
                                          'Popularity': row['popularity_score'],
                                          'Vectors': get_vec_as_list(i, merged_df),
                                          'Labels': get_labels_as_list(row['article_id'], labels_df),
                                          'Country': row['country'],
                                          'x': row['x'],
                                          'y': row['y']}
    return article_data


def generate_json(map_directory, article_data):
    #print(len(article_data))
    with open(map_directory + 'domain.json', 'w') as outfile:
        json.dump(article_data, outfile)
    return


def main(map_directory):
    list_of_results = list_results(map_directory)
    merged_article_df = merge_results(list_of_results)
    article_data = create_list_article_data(merged_article_df, map_directory)
    generate_json(map_directory, article_data)


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
