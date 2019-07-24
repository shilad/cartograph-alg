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
import argparse


def create_merged_df(map_directory, experiment_directory, country_labels, cluster_groups, embeddings):
    domain_concepts_df = pd.read_csv(map_directory+'/domain_concept.csv')
    pop_scores_df = pd.read_csv(map_directory + '/popularity_score.csv')
    xy_df = pd.read_csv(experiment_directory + embeddings)

    country_label_df = pd.read_csv(experiment_directory + country_labels)
    cluster_groups_df = pd.read_csv(experiment_directory + cluster_groups)
    countries_df = pd.merge(cluster_groups_df, country_label_df, on='country')
    countries_df = countries_df.drop(columns=['country'])
    if 'label_name' in countries_df.columns:
        countries_df = countries_df.rename(columns={'label_name': 'country'})
    else:
        countries_df = countries_df.rename(columns={'0': 'country'})

    df_list = [domain_concepts_df, pop_scores_df, countries_df, xy_df]
    df = reduce(lambda df1, df2: pd.merge(df1, df2, on='article_id'), df_list) #reduce(pd.merge, df_list)
    return df


def create_list_article_data(merged_df, method):
    article_data = {}
    for i, row in merged_df.iterrows():
        article_data[row['article_id']] = {'Article': row['article_name'],
                                          'Popularity': row['popularity_score'],
                                          'Country': row['country'],
                                          'x': row['x'],
                                          'y': row['y']}
        if method == 'noise':
            article_data[row['article_id']]['relevance'] = row['distance']

    return article_data


def generate_json(experiment_directory, article_data, output_name):
    with open(experiment_directory + output_name, 'w') as outfile:
        json.dump(article_data, outfile)
    return


def main(map_directory, experiment_directory, method, country_labels, cluster_groups, embeddings, output_name):
    merged_article_df = create_merged_df(map_directory, experiment_directory, country_labels, cluster_groups, embeddings)
    article_data = create_list_article_data(merged_article_df, method)
    generate_json(experiment_directory, article_data, output_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_directory', required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--filter_method', required=True)
    parser.add_argument('--country_labels', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--embeddings', required=True)
    parser.add_argument('--output_name', required=True)

    args = parser.parse_args()
    main(args.map_directory,
         args.experiment,
         args.filter_method,
         args.country_labels,
         args.cluster_groups,
         args.embeddings,
         args.output_name)
