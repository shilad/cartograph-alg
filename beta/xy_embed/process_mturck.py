import pandas as pd
import numpy as np
from gensim.parsing.porter import PorterStemmer

hit_labels = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/hit_labels.csv")
hit_articles = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/hit_articles.csv")

hit_labels = hit_labels[['project', 'cluster_alg', 'cluster_num', 'label_name', 'avg_borda']]
hit_articles = hit_articles[['project', 'cluster_alg', 'cluster_num', 'article_name']]

merged = pd.merge(hit_articles, hit_labels, on=['cluster_num', 'cluster_alg', 'project'])
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


def add_country_label_counts(labels_df):
    """Add counts per label, country pair to the data frame as "country_label_count", counting how many times a label """
    counts = labels_df.groupby(['cluster_num', 'project', 'label_name']).size().reset_index(name="country_label_count")
    return pd.merge(labels_df, counts, on=('cluster_num', 'label_name', 'project'))


def add_label_counts(labels_df):
    """Add counts per label to the data frame as "label_count", counting how many times a label appears selected for a specific project"""
    counts = labels_df.groupby(["label_name"]).size().reset_index(name="label_count")
    return pd.merge(labels_df, counts, on=["label_name" ])


def add_country_counts(labels_df, articles_df):
    """Add number of total labels and articles per country to the data frame as "country_count" ."""
    counts = labels_df.groupby(["cluster_num", "project"]).size().reset_index(name="num_country_labels")
    labels_df = pd.merge(labels_df, counts, on=["cluster_num", "project"])
    counts = articles_df.groupby(["cluster_num", "project"])['article_name'].nunique().reset_index(name="num_country_articles")
    return pd.merge(labels_df, counts, on=["cluster_num", "project"])


def add_totals(labels_df, articles_df):
    counts = articles_df.groupby(['project'])['cluster_num'].nunique().reset_index(name="num_countries")
    labels_df = pd.merge(labels_df, counts, on=["project"])
    counts = articles_df.groupby(['project'])['article_name'].nunique().reset_index(name="num_articles")
    labels_df = pd.merge(labels_df, counts, on=["project"])
    return labels_df


def add_tfidf_scores(labels_df):
    """Creates a tf-idf column for the data frame"""
    tf = (labels_df['country_label_count'] / (labels_df['num_country_labels'] + 1))
    idf = np.log(labels_df['num_articles'] / (labels_df['label_count'] + 10))
    labels_df['tf'] = tf
    labels_df['idf'] = idf
    labels_df['tfidf'] = tf * idf
    return labels_df


algs = ["kmeans_plain", "kmeans_augmented"]
projects = ["food", "internet", "technology", "media"]
label_sources = ["h_cat", "key_phrases", "key_words", "lda", "links"]
labels = {}
labels.update({'h_cat': [], 'key_words':[], 'key_phrases': [], 'links': [], 'lda': []})
for alg in algs:
    for project in projects:
        for label_source in label_sources:
            if(alg == "LDA"):
                top_labels = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study-old/" + project + "/" + alg + "/labels/" + "LDA_labels" + "/top_labels.csv")
            else:
                top_labels = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study-old/" + project + "/" + alg + "/labels/" + label_source + "/top_labels.csv")
                top_labels = pd.DataFrame(top_labels)
                if "label" in top_labels.columns:
                    tops = top_labels['label']
                if "label_name" in top_labels.columns:
                    tops = top_labels['label_name']
                labels[label_source].extend(tops.values)

print(labels.get('links'))

def main(hit_labels, hit_articles, labels):

    # Calculate tf-idf scores
    hit_labels = add_country_label_counts(hit_labels)
    hit_labels = add_label_counts(hit_labels)
    hit_labels = add_country_counts(hit_labels, hit_articles)
    hit_labels = add_totals(hit_labels, hit_articles)
    hit_labels = add_tfidf_scores(hit_labels)
    top_df = hit_labels[['cluster_num', 'label_name', 'tfidf', 'project', 'cluster_alg', 'avg_borda']]
    top_df['label_name'] = top_df['label_name'].str.replace('_', ' ')

    for key in labels.keys():
        for index, row in top_df.iterrows():
            if row.label_name in labels.get(key):
                top_df.loc[index, key] = 1
    final = top_df.fillna(0)
    final['h_cat_tfidf'] = final['h_cat'] * final['tfidf']
    final['key_words_tfidf'] = final['key_words'] * final['tfidf']
    final['key_phrases_tfidf'] = final['key_phrases'] * final['tfidf']
    final['links_tfidf'] = final['links'] * final['tfidf']
    final['lda_tfidf'] = final['lda'] * final['tfidf']
    print(final[final['label_name'] == 'celebrity chef'])

    final.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/beta/regression/final_labels.csv")

main(hit_labels, hit_articles, labels)
