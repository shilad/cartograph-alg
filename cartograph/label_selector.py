"""
Given two data frames (one containing article id's and label id's, and one including article id's and country clusters),
outputs a list countries and their labels.
Author: Lily Irvin
"""
import argparse

import pandas as pd
from collections import defaultdict
import math
import operator
from nltk.stem import PorterStemmer


def add_countries(article_labels_csv, country_clusters_csv):
    """Appends a new column with country assignments to the article_labels dataframe."""

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
        tfidf_scores[row['country']][row['label_id']] = country_label_counts[row['country']][row['label_id']] / \
                                                        math.log(total_counts[row['label_id']] + 10.0)

    return tfidf_scores


def assign_country_label_ids(label_names_df, tfidf_scores, num_countries):
    """Output: Dictionary --> key = country, value = label"""
    ps = PorterStemmer()
    country_labels = {}
    for i in range(num_countries):
        print('---')

        # Fix this to choose the higher TFIDF score, not just the first one to be entered:
        while i not in country_labels:
            label_id = max(tfidf_scores[i].items(), key=operator.itemgetter(1))[0]
            label = label_names_df.iloc[label_id].loc['label']
            duplicates = False
            for value in country_labels.values():
                if ps.stem(label) == ps.stem(value):
                    duplicates = True
            if duplicates:
                del tfidf_scores[i][label_id]
            else:
                country_labels[i] = label

        top_five = sorted(tfidf_scores[i].items(), key=operator.itemgetter(1), reverse=True)[:5]
        for i in top_five:
            print(label_names_df.iloc[i[0]].loc['label'], i[1])

    return country_labels


def main(experiment_dir, article_to_label_path, label_name_path):
    # Read in labels datasets
    labels_df = add_countries(article_to_label_path, experiment_dir + '/cluster_groups.csv')
    label_names_df = pd.read_csv(label_name_path)

    # Calculate tf-idf scores
    num_countries = get_num_countries(experiment_dir + '/cluster_groups.csv')
    country_label_counts = get_country_label_counts(labels_df, num_countries)
    total_counts = get_total_counts(labels_df)
    tfidf_scores = get_tfidf_scores(labels_df, country_label_counts, total_counts, num_countries)

    # Assign labels
    country_labels = assign_country_label_ids(label_names_df, tfidf_scores, num_countries)

    # Create results data frame
    df = pd.DataFrame(country_labels,  index=[0]).T
    df['country'] = df.index
    df.to_csv(experiment_dir + '/country_labels.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)

    args = parser.parse_args()

    main(args.experiment, args.articles_to_labels, args.label_names)
