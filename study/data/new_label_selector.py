"""
Given cluster groups, article labels, produces a dataframe with label tfidf scores.
"""

import argparse
import numpy as np
import pandas as pd
from gensim.parsing.porter import PorterStemmer
BLACK_LIST = ['food', 'foods', 'food and drink', 'media', 'internet', 'technology']


def get_labels_df(article_labels, cluster_groups, label_names):
    article_labels = pd.merge(article_labels, cluster_groups, on='article_id')
    article_labels = pd.merge(article_labels, label_names, on='label_id')
    return article_labels


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
    top_labels = [[] for x in range(country_scores['num_countries'][0])]
    used_stems = set()
    country_scores = country_scores.fillna(0)
    for row in country_scores.itertuples():
        if row.stem not in used_stems and row.stem is not int:
            # selecting top 20 labels
            if len(top_labels[row.country]) < 90:
                top_labels[row.country].extend([str(row.label).lower().replace(' ', '_').strip(), float(row.tfidf), row.country])
                used_stems.add(row.stem)
                if(row.label == "preparation"):
                    print(row.label)
                    print("Ccccccccccccccccc")
    return top_labels


def get_country_label_matrix(article_labels):
    article_labels = add_country_label_counts(article_labels)
    article_labels = add_label_counts(article_labels)
    article_labels = add_country_counts(article_labels)
    article_labels = add_totals(article_labels)
    article_labels = add_tfidf_scores(article_labels)
    country_labels = article_labels.drop(columns=['article_id']).drop_duplicates()
    return country_labels


def get_top_labels_alg(article_labels, label_source, alg, project):
    country_labels = get_country_label_matrix(article_labels)
    top_labels = get_top_labels(country_labels)
    top = [item for sublist in top_labels for item in sublist]
    top = np.reshape(top, (-1,3))
    column_names = ["label_name", "tfidf", "cluster_num"]
    top_labels_df = pd.DataFrame(top, columns=column_names)
    top_labels_df[label_source] = int(1)
    top_labels_df["cluster_alg"] = alg
    top_labels_df['project'] = project
    top_labels_df.tfidf = top_labels_df.tfidf.astype(float)
    return top_labels_df


def main(label_dir, cluster_groups, topic):

    label_types = ["h_cat", "keyphrases", "keywords", "links"]
    article_label_csv = ["article_hierarchical_categories.csv", "article_keyphrases.csv", "article_keywords.csv","article_links.csv"]
    label_name_csv = ["hierarchical_category_names.csv", "keyphrases_names.csv", "keyword_names.csv",  "link_names.csv"]
    algs = ["kmeans_plain", "kmeans_augmented"]

    top_labels = []
    for i in range(len(algs)):
        print(algs[i])
        cluster_groups_df = pd.read_csv(cluster_groups + algs[i] + "_cluster_groups.csv")
        for j in range(len(label_types)):
            article_labels = pd.read_csv(label_dir + article_label_csv[j], encoding='utf-8', engine='python')
            label_names = pd.read_csv(label_dir + label_name_csv[j])

            article_labels_df = get_labels_df(article_labels, cluster_groups_df, label_names)
            label_df = get_top_labels_alg(article_labels_df, label_types[j], algs[i], topic)
            top_labels.append(label_df)

    top_labels_df = pd.concat(top_labels)
    top_labels_df = top_labels_df.fillna(0)
    # top_labels_df.to_csv(args.output_file)
    return top_labels_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--label_dir', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--output_file', required=True)

    args = parser.parse_args()
    topics = ['Media', 'food', 'technology', 'internet']
    final_labels_features = []
    for i in range(len(topics)):
        print(topics[i])
        label_dir = args.label_dir + topics[i] + "/"
        topic_label_top = main(label_dir, args.cluster_groups, topics[i])
        final_labels_features.append(topic_label_top)

    top_labels_df = pd.concat(final_labels_features)
    top_labels_df = top_labels_df.fillna(0)
    # print(top_labels_df)
    top_labels_df["h_cat_tfidf"] = top_labels_df['h_cat'] * top_labels_df['tfidf']
    top_labels_df["links_tfidf"] = top_labels_df['links'] * top_labels_df['tfidf']
    top_labels_df["key_words_tfidf"] = top_labels_df['keywords'] * top_labels_df['tfidf']
    top_labels_df["key_phrases_tfidf"] = top_labels_df['keyphrases'] * top_labels_df['tfidf']

    print(top_labels_df)

    top_labels_df = top_labels_df[['project',  "cluster_alg",  'label_name', 'tfidf', 'h_cat_tfidf', 'links_tfidf', 'key_words_tfidf', 'key_phrases_tfidf']]
    top_labels_df = top_labels_df.fillna(0)
    # print(top_labels_df[top_labels_df['h_cat_tfidf'] > 0])
    top_labels_df = top_labels_df.groupby(['label_name', 'project', 'cluster_alg'], as_index=False).sum()
    print(top_labels_df)

pd.set_option('display.max_columns', None)
top_labels_df = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/feature/top_label_tfidfs.csv")
hit_labels = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/hit_labels.csv")
hit_labels = hit_labels[['name', 'cluster_alg', 'avg_borda', 'share', 'project' ]]
hit_labels = hit_labels[hit_labels['project'] == "food"]
top_labels_df = top_labels_df[top_labels_df['project'] == "food"]
top_labels_df = top_labels_df[top_labels_df['cluster_alg'] != "LDA" ]
training_data = pd.merge(hit_labels, top_labels_df, how='outer', indicator=True, on=['name', 'cluster_alg', 'project'])
tt = training_data.query('_merge == "left_only"')
print(tt)
top_labels_df.to_csv(args.output_file)
# training_data.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/feature/trainingdata.csv")
#
