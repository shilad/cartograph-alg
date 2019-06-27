"""
Given two data frames (one containing article id's and label id's, and one including article id's and country clusters),
outputs a list countries and their labels.
Author: Lily Irvin
"""
import argparse

import numpy as np
import pandas as pd
import operator
from nltk.stem import PorterStemmer



def add_country_label_counts(labels_df):
    """Add counts per label, country pair to the data frame as "country_label_count" ."""
    counts = labels_df.groupby(["country", "label_id"]).size().reset_index(name="country_label_count")
    return pd.merge(labels_df, counts, on=('country', 'label_id'))


def add_label_counts(labels_df):
    """Add counts per label to the data frame as "label_count" ."""
    counts = labels_df.groupby(["label_id"]).size().reset_index(name="label_count")
    return pd.merge(labels_df, counts, on='label_id')


def add_country_counts(labels_df):
    """Add number of total labels and articles per country to the data frame as "country_count" ."""
    counts = labels_df.groupby(["country"]).size().reset_index(name="num_country_labels")
    labels_df = pd.merge(labels_df, counts, on='country')
    counts = labels_df.groupby(["country"])['article_id'].nunique().reset_index(name="num_country_articles")
    return pd.merge(labels_df, counts, on='country')

def add_totals(labels_df):
    labels_df['num_countries'] = labels_df['country'].nunique()
    labels_df['num_articles'] = labels_df['article_id'].nunique()
    return labels_df


def add_pmi(labels_df):
    """Creates a pmi column for the data frame"""
    tf = (labels_df['country_label_count'] / (labels_df['num_country_labels'] + 1))
    idf = (labels_df['num_articles'] / (labels_df['label_count'] + 0.25 * labels_df['num_country_articles']))

    labels_df['tf'] = tf
    labels_df['idf'] = idf
    labels_df['tfidf'] = tf * idf

    return labels_df


def add_tfidf_scores(labels_df):
    """Creates a tf-idf column for the data frame"""
    tf = (labels_df['country_label_count'] / (labels_df['num_country_labels'] + 1))
    idf = np.log(labels_df['num_articles'] / (labels_df['label_count'] + 10))

    labels_df['tf'] = tf
    labels_df['idf'] = idf
    labels_df['tfidf'] = tf * idf

    return labels_df


def assign_country_label_ids(label_names_df, tfidf_scores, num_countries, country_label_counts, total_counts, labels_df):
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
            print(label_names_df.iloc[i[0]].loc['label'], i[1], 'tf '
                  + str(country_label_counts[labels_df['country'].iloc[i[0]]][i[0]]), "idf: "
                  + str(total_counts[i[0]]))
    return country_labels


def main(experiment_dir, article_labels, percentile):
    # choose the best percentile labels
    if 'distance' in article_labels.columns:
        mask = article_labels['distance'] < article_labels['distance'].quantile(float(percentile))
        article_labels = article_labels[mask]

    # Calculate tf-idf scores
    article_labels = add_country_label_counts(article_labels)
    article_labels = add_label_counts(article_labels)
    article_labels = add_country_counts(article_labels)
    article_labels = add_totals(article_labels)
    tfidf_scores = add_tfidf_scores(article_labels)

    country_tf_idf = tfidf_scores.drop(columns=['article_id', 'distance']).drop_duplicates()

    print(country_tf_idf.groupby('country').apply(lambda x: x.nlargest(5, 'tfidf')).reset_index(drop=True))


    # Assign labels
    # country_labels = assign_country_label_ids(label_names_df, tfidf_scores, num_countries, country_label_counts, total_counts, labels_df)
    #
    # # Create results data frame
    # df = pd.DataFrame(country_labels,  index=[0]).T
    # df['country'] = df.index
    # df.to_csv(experiment_dir + '/country_labels.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--percentile', required=True)


    args = parser.parse_args()

    article_labels = pd.read_csv(args.articles_to_labels)
    country_clusters = pd.read_csv(args.experiment + '/cluster_groups.csv')
    label_names = pd.read_csv(args.label_names)
    article_labels = pd.merge(article_labels, country_clusters, on='article_id')
    article_labels = pd.merge(article_labels, label_names, on='label_id')

    main(args.experiment, article_labels, args.percentile)
