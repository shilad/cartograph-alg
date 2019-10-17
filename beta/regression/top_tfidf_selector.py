"""
Clean ml traning data.

Author: Lu Li
"""

import pandas as pd
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from gensim.parsing.porter import PorterStemmer
BLACK_LIST = ['food', 'foods', 'food and drink', 'media', 'internet', 'technology']


def add_country_label_counts(labels_df):
    print(labels_df)
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


def get_top_labels(country_scores):
    """Output: Dictionary --> key = country, value = list of top labels"""
    ps = PorterStemmer()
    country_scores['stem'] = ps.stem_documents([str(word) for word in country_scores['label']])
    country_scores = country_scores.sort_values(by="tfidf", ascending=False)
    country_scores_pmi = country_scores.sort_values(by="pmi", ascending=False)
    top_labels = [[] for x in range(country_scores['num_countries'][0])]
    top_labels_pmi = [[] for x in range(country_scores_pmi['num_countries'][0])]

    used_stems = set()
    used_stems_pmi = set()

    for row in country_scores.itertuples():
        if row.stem not in used_stems:
            if len(top_labels[row.country]) < 40:
                top_labels[row.country].extend([row.label.lower().replace('_', ' ').strip(), row.tfidf, row.pmi, row.country])
                used_stems.add(row.stem)

    for row in country_scores_pmi.itertuples():
        if row.stem not in used_stems_pmi:
            if len(top_labels_pmi[row.country]) < 40:
                top_labels_pmi[row.country].extend([row.label.lower().replace('_', ' ').strip(), row.tfidf, row.pmi, row.country])
                used_stems_pmi.add(row.stem)

    return top_labels, top_labels_pmi


def main(experiment_dir, article_labels, percentile, label_source, output_file):
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
    if 'distance' in article_labels.columns:
        country_labels = article_labels.drop(columns=['article_id', 'distance']).drop_duplicates()
    else:
        country_labels = article_labels.drop(columns=['article_id']).drop_duplicates()

    top_tfidf, top_pmi = get_top_labels(country_labels)
    top = [item for sublist in top_tfidf for item in sublist]
    top_pmi = [item for sublist in top_pmi for item in sublist]

    top = np.reshape(top, (-1,4))
    top_pmi = np.reshape(top_pmi, (-1,4))

    column_names = ["label", "tfidf", "pmi", "country"]
    top = pd.DataFrame(top, columns=column_names)
    top_pmi = pd.DataFrame(top_pmi, columns=column_names)
    top = pd.merge(top, top_pmi, how="outer")
    top[label_source] = 1

    top.to_csv(experiment_dir + "/" + output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--percentile', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--label_name', required=True)
    parser.add_argument('--label_source', required=True)
    parser.add_argument('--output_file', required=True)
    args = parser.parse_args()

    article_labels = pd.read_csv(args.articles_to_labels, encoding='utf-8',  engine='python')
    country_clusters = pd.read_csv(args.experiment + args.cluster_groups)
    article_labels = pd.merge(article_labels, country_clusters, on='article_id')

    label_names = pd.read_csv(args.label_name)
    article_labels = pd.merge(article_labels, label_names, on='label_id')

    main(args.experiment, article_labels, args.percentile, args.label_source, args.output_file)