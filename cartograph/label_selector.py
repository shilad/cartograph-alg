"""
Given two data frames (one containing article id's and label id's, and one including article id's and country clusters),
outputs a list countries and their labels.
Author: Lily Irvin
"""
import argparse
import numpy as np
import pandas as pd
from gensim.parsing.porter import PorterStemmer
from collections import defaultdict


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

    labels_df['pmi'] = tf * idf

    return labels_df


def add_tfidf_scores(labels_df):
    """Creates a tf-idf column for the data frame"""
    tf = (labels_df['country_label_count'] / (labels_df['num_country_labels'] + 1))
    idf = np.log(labels_df['num_articles'] / (labels_df['label_count'] + 10))

    labels_df['tf'] = tf
    labels_df['idf'] = idf
    labels_df['tfidf'] = tf * idf

    return labels_df


def assign_country_label_ids(country_scores, label_score):
    """Output: Dictionary --> key = country, value = label"""

    ps = PorterStemmer()
    country_scores['stem'] = ps.stem_documents([str(word) for word in country_scores['label']])
    country_scores = country_scores.sort_values(by=label_score, ascending=False)
    final_labels = {}
    used_stems = set()

    for row in country_scores.itertuples():
        if row.country not in final_labels and row.stem not in used_stems:
            final_labels[row.country] = row.label.replace('_', ' ')
            used_stems.add(row.stem)

    return final_labels


def get_top_labels(country_scores, label_score):
    """Output: Dictionary --> key = country, value = list of top labels"""

    ps = PorterStemmer()
    country_scores['stem'] = ps.stem_documents([str(word) for word in country_scores['label']])
    country_scores = country_scores.sort_values(by=label_score, ascending=False)
    top_labels = [defaultdict(str) for x in range(country_scores['num_countries'][0])]
    used_stems = set()

    for row in country_scores.itertuples():
        if row.stem not in used_stems:
            if len(top_labels[row.country]) < 10:
                top_labels[row.country][row.label_id] = row.label.replace('_', ' ')
                used_stems.add(row.stem)

    return top_labels


def main(experiment_dir, article_labels, percentile, label_score):
    # choose the best percentile labels
    if 'distance' in article_labels.columns:
        mask = article_labels['distance'] < article_labels['distance'].quantile(float(percentile))
        article_labels = article_labels[mask]

    # Calculate tf-idf scores
    article_labels = add_country_label_counts(article_labels)
    article_labels = add_label_counts(article_labels)
    article_labels = add_country_counts(article_labels)
    article_labels = add_totals(article_labels)
    article_labels = add_tfidf_scores(article_labels)
    article_labels = add_pmi(article_labels)

    country_labels = article_labels.drop(columns=['article_id', 'distance']).drop_duplicates()

    final_labels = assign_country_label_ids(country_labels, label_score)

    # # Create results data frame
    df = pd.DataFrame(final_labels,  index=[0]).T
    df['country'] = df.index
    df.to_csv(experiment_dir + '/country_labels.csv', index=True)

    # # Get top label candidates
    top = get_top_labels(country_labels, label_score)
    print(top)

    # top_df = pd.DataFrame.from_dict(top, orient='index')
    # top_df['country'] = top_df.set_index
    # top_df.to_csv(experiment_dir + '/top_label_candidates.csv', index=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--percentile', required=True)
    parser.add_argument('--label_score', required=True)

    args = parser.parse_args()

    article_labels = pd.read_csv(args.articles_to_labels)
    country_clusters = pd.read_csv(args.experiment + '/cluster_groups.csv')
    label_names = pd.read_csv(args.label_names)
    article_labels = pd.merge(article_labels, country_clusters, on='article_id')
    article_labels = pd.merge(article_labels, label_names, on='label_id')

    main(args.experiment, article_labels, args.percentile, args.label_score)
