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
def assign_country_label_ids(country_scores, label_score, num_candidates, use_label_candidates):
    """Output: Dictionary --> key = country, value = label"""
    ps = PorterStemmer()
    country_scores['stem'] = ps.stem_documents([str(word) for word in country_scores['label']])
    country_scores = country_scores.sort_values(by=label_score, ascending=False)
    used_stems = set()
    if use_label_candidates == True:
        final_labels = defaultdict(set)
        final_ids = defaultdict(set)
        for row in country_scores.itertuples():
            if len(final_labels[row.country]) <= num_candidates and row.stem not in used_stems and row.stem not in BLACK_LIST:
                final_labels[row.country].add([row.label.lower().replace('_', ' ').strip(), row.tfidf, row.pmi])
                final_ids[row.country].add(int(row.label_id))
                used_stems.add(row.stem)
    else:
        final_labels = {}
        final_ids = {}
        for row in country_scores.itertuples():
            if row.country not in final_labels and row.stem not in used_stems and row.stem not in BLACK_LIST:
                final_labels[row.country] = [row.label.lower().replace('_', ' ').strip(), row.tfidf, row.pmi]
                final_ids[row.country] = row.label_id
                used_stems.add(row.stem)
    return final_labels, final_ids
def get_top_labels(country_scores, label_score):
    """Output: Dictionary --> key = country, value = list of top labels"""
    ps = PorterStemmer()
    country_scores['stem'] = ps.stem_documents([str(word) for word in country_scores['label']])
    country_scores = country_scores.sort_values(by=label_score, ascending=False)
    top_labels = [[] for x in range(country_scores['num_countries'][0])]
    used_stems = set()
    for row in country_scores.itertuples():
        if row.stem not in used_stems:
            if len(top_labels[row.country]) < 30:
                top_labels[row.country].extend([row.label.lower().replace('_', ' ').strip(), row.tfidf, row.pmi])
                used_stems.add(row.stem)
    return top_labels
def main(experiment_dir, article_labels, percentile, label_score, output_file, use_label_candidates, num_candidates, purpose, label_path, alg, label_source):
    # choose the best percentile labels
    if 'distance' in article_labels.columns:
        # print("Selecting labels with noise filtering------------------------------")
        mask = article_labels['distance'] < article_labels['distance'].quantile(float(percentile))
        article_labels = article_labels[mask]
    # Calculate tf-idf scores
    article_labels = add_country_label_counts(article_labels)
    article_labels = add_label_counts(article_labels)
    article_labels = add_country_counts(article_labels)
    article_labels = add_totals(article_labels)
    article_labels = add_tfidf_scores(article_labels)
    tf_idf_score = article_labels[['article_id', 'label_id', 'tfidf']]
    article_labels = add_pmi(article_labels)
    scores = article_labels[['label_id', 'tfidf','pmi']]
    scores = pd.DataFrame(scores)
    if 'distance' in article_labels.columns:
        country_labels = article_labels.drop(columns=['article_id', 'distance']).drop_duplicates()
    else:
        country_labels = article_labels.drop(columns=['article_id']).drop_duplicates()
    final_labels, final_scores = assign_country_label_ids(country_labels, label_score, num_candidates, use_label_candidates)
    # # Create results data frame
    df = pd.DataFrame.from_dict(final_labels,  orient='index', columns=['label_name', 'tfidf', 'pmi'])
    df['country'] = df.index
    if purpose == 'study':
        df = df.set_index('country')
    if alg == "LDA":
        df.to_csv(experiment_dir + "/labels/LDA_labels" + '/final_labels.csv', index=True)
    else:
        df.to_csv(label_path + '/final_labels.csv', index=True)
    # else:
    #     df['label_id'] = np.array(list(final_scores.values())).T
    #     df.columns = ['label_name', 'country', 'label_id']
    #     df.to_csv(experiment_dir + output_file, index=True)
    # # Get top label candidates
    top = get_top_labels(country_labels, label_score)
    top = [item for sublist in top for item in sublist]
    top = np.reshape(top, (-1,3))
    column_names = ["label", "tfidf", "pmi"]
    top = pd.DataFrame(top, columns=column_names)
    if label_source == "h_cat":
        top['h_cat'] = 1
        top['key_phrases'] = 0
        top['key_words'] = 0
        top['lda_label'] = 0
        top['links'] = 0
    if label_source == "key_phrases":
        top['h_cat'] = 0
        top['key_phrases'] = 1
        top['key_words'] = 0
        top['lda_label'] = 0
        top['links'] = 0
    if label_source == "key_words":
        top['h_cat'] = 0
        top['key_phrases'] = 0
        top['key_words'] = 1
        top['lda_label'] = 0
        top['links'] = 0
    if label_source == "lda":
        top['h_cat'] = 0
        top['key_phrases'] = 0
        top['key_words'] = 0
        top['lda_label'] = 1
        top['links'] = 0
    if label_source == "links":
        top['h_cat'] = 0
        top['key_phrases'] = 0
        top['key_words'] = 0
        top['lda_label'] = 0
        top['links'] = 1
    top['cluster_alg'] = alg
    if alg == "kmeans_augmented":
        top['augmented'] = 1
        top['LDA'] = 0
        top['plain'] = 0
    if alg == "kmeans_plain":
        top['augmented'] = 0
        top['LDA'] = 0
        top['plain'] = 1
    if alg == "LDA":
        top['augmented'] = 0
        top['LDA'] = 1
        top['plain'] = 0
    if alg == "LDA" and label_source == "lda":
        top.to_csv(experiment_dir + "/labels/"  + "LDA_labels"  + '/top_labels.csv')
    else:
        top.to_csv(experiment_dir + "/labels/" + label_source +'/top_labels.csv')
        print("---------------------------------------" + experiment_dir + "/labels/" + label_source +'/top_labels.csv')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--percentile', required=True)
    parser.add_argument('--label_score', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--use_label_candidates', required=True)
    parser.add_argument('--num_candidates', required=False, type=int)
    parser.add_argument('--purpose', required=True)
    parser.add_argument('--label_path', required=True)
    parser.add_argument('--alg', required=True)
    parser.add_argument('--label_source', required=True)
    args = parser.parse_args()
    article_labels = pd.read_csv(args.articles_to_labels, encoding='utf-8',  engine='python')
    country_clusters = pd.read_csv(args.experiment + args.cluster_groups )
    label_names = pd.read_csv(args.label_names )
    article_labels = pd.merge(article_labels, country_clusters, on='article_id')
    article_labels = pd.merge(article_labels, label_names, on='label_id')
    main(args.experiment, article_labels, args.percentile, args.label_score, args.output_file, args.use_label_candidates, args.num_candidates, args.purpose, args.label_path, args.alg, args.label_source)