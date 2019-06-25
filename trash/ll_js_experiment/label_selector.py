"""
Given two data frames (one containing article id's and label id's, and one including article id's and country clusters),
outputs a list countries and their labels.

Author: Lily Irvin
"""


import pandas as pd
from collections import defaultdict
import math
import operator


def add_countries(article_labels_csv, country_clusters_csv):
    """Creates a new data frame including both article labels and article country assignments."""

    article_labels = pd.read_csv(article_labels_csv)
    country_clusters = pd.read_csv(country_clusters_csv)
    article_labels = pd.merge(article_labels, country_clusters, on='article_id')
    return article_labels


def get_num_countries(country_clusters_csv):
    country_clusters = pd.read_csv(country_clusters_csv)
    return max(country_clusters['country']) + 1


def get_country_label_counts(labels_df, num_countries):
    """Output: List of default dictionaries (one per country) --> key = label id, value = number of times that label
               appears in that country"""

    country_label_counts = [defaultdict(int) for x in range(num_countries)]
    for index, row in labels_df.iterrows():
        country_label_counts[row['country']][row['label_id']] += 1
    return country_label_counts


def get_total_counts(labels_df):
    """Output: Default dictionary --> key = label id, value = number of times that label appears in the domain
               concept"""

    total_counts = defaultdict(int)
    for index, row in labels_df.iterrows():
        total_counts[row['label_id']] += 1
    return total_counts


def get_tfidf_scores(labels_df, country_label_counts, total_counts, num_countries):
    """Output: List of default dictionaries (one per country) --> key = label id, value = TF-IDF score for that label
               in that country"""

    tfidf_scores = [defaultdict(int) for x in range(num_countries)]
    for index, row in labels_df.iterrows():
        tfidf_scores[row['country']][row['label_id']] = math.log(country_label_counts[row['country']][row['label_id']] + 1.0) / \
                                                        math.log(total_counts['row_id'] + 10.0)
    return tfidf_scores


def assign_country_label_ids(tfidf_scores, num_countries):
    """Output: Dictionary --> key = country, value = label"""

    country_labels = {}
    for i in range(num_countries):
        label_id = max(tfidf_scores[i].items(), key=operator.itemgetter(1))[0]
        country_labels[i] = get_label(label_id, map_directory)

    return country_labels


def get_label(label_id, map_directory):
    label_ids = pd.read_csv(map_directory + '/category_names.csv')
    return label_ids.iloc[label_id].loc['label']


def create_csv(country_labels, directory):
    df = pd.DataFrame(country_labels, index=[0]).T
    df['country'] = df.index
    df.to_csv(directory + '/country_labels.csv', index=True)


def main(map_directory):
    labels_df = add_countries(map_directory + '/article_categories.csv',
                              map_directory + '/cluster_groups.csv')
    num_countries = get_num_countries(map_directory + '/cluster_groups.csv')
    country_label_counts = get_country_label_counts(labels_df, num_countries)
    total_counts = get_total_counts(labels_df)
    tfidf_scores = get_tfidf_scores(labels_df, country_label_counts, total_counts, num_countries)
    country_labels = assign_country_label_ids(tfidf_scores, num_countries)
    create_csv(country_labels, map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)

