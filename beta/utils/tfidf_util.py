"""
Calculate TF-IDF of (top) label candidates given article_labels and cluster_groups
"""
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
    projects = ['food', 'internet', 'technology', 'media']
    final = []
    for project in projects:
        final.append(pd.read_csv(
            "data/" + project + "/hierarchical_category_names.csv"))
        final.append(
            pd.read_csv("data/" + project + "/keyphrases_names.csv"))
        final.append(pd.read_csv("data/" + project + "/keyword_names.csv"))
        final.append(pd.read_csv("data/" + project + "/lda_label_names.csv"))
        final.append(pd.read_csv("data/" + project + "/link_names.csv"))
        final.append(pd.read_csv("data/" + project + "/lda_label_names.csv"))


    final = pd.concat(final)
    counts = final.groupby(["label"]).size().reset_index(name="label_count")
    counts2 = labels_df.groupby(["label"]).size().reset_index(name="label_count_project")
    with_count_all = pd.merge(labels_df, counts, on='label')
    return pd.merge(with_count_all, counts2, on='label')


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


def get_top_labels(label_article_cluster):
    ps = PorterStemmer()
    label_article_cluster['stem'] = ps.stem_documents([str(word) for word in label_article_cluster['label']])
    label_article_cluster = label_article_cluster.sort_values(by="tfidf", ascending=False)
    top_labels = [[] for x in range(label_article_cluster['num_countries'][0])]
    used_stems = set()
    print(label_article_cluster)
    for row in label_article_cluster.itertuples():
        if row.stem not in used_stems and row.stem not in BLACK_LIST:
            if len(top_labels[row.country]) < 200:
                # cluster_id, label_name, tfidf
                top_labels[row.country].extend([row.country, row.label.lower().replace(' ', '_').strip(), row.tfidf, row.country_label_count, row.num_country_labels, row.num_articles, row.label_count, row. label_count_project, row.idf, row.tf])
                used_stems.add(row.stem)

    top_labels_df = pd.DataFrame(np.reshape(top_labels, (-1, 10)), columns=["country", "label_name", "tfidf", "country_label_count", 'num_country_labels', "num_articles", "label_count", "label_count_project", "idf", "tf"])
    return top_labels_df


def calc_tfidf(article_labels, cluster_groups, selected_columns):
    # article id, label id, cluster group
    label_article_cluster = pd.merge(article_labels, cluster_groups, on='article_id')
    label_article_cluster = add_country_label_counts(label_article_cluster)
    label_article_cluster = add_label_counts(label_article_cluster)
    label_article_cluster = add_country_counts(label_article_cluster)
    label_article_cluster = add_totals(label_article_cluster)
    # label_article_cluster['label_cnt'] = label_article_cluster['label_count_project'] - label_article_cluster['country_label_count']

    tf = (label_article_cluster['country_label_count'] / (label_article_cluster['num_country_labels'] + 1))
    idf = label_article_cluster['num_countries'] / (label_article_cluster['label_count'])

    label_article_cluster['tf'] = tf
    label_article_cluster['idf'] = idf
    label_article_cluster['tfidf'] = tf * idf
    return label_article_cluster[selected_columns]

