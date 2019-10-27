from gensim.parsing.porter import PorterStemmer
import beta.utils.tfidf_util as utils
import numpy as np
import pandas as pd
import argparse


def main(experiment_dir, article_labels, cluster_groups, label_source, output_file):
    tfidf = utils.calc_tfidf(article_labels, cluster_groups, ['article_id', 'label_id', 'label', 'tfidf', 'num_countries', 'country', 'country_label_count', 'num_country_labels', 'num_articles', 'label_count', 'label_count_project', 'tf', 'idf'])
    top_labels = utils.get_top_labels(tfidf)
    top_labels[label_source] = 1
    top_labels.to_csv(experiment_dir + "/" + output_file)
    # country_labels = article_labels.drop(columns=['article_id']).drop_duplicates()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--label_source', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--output_file', required=True)

    args = parser.parse_args()

    article_labels = pd.read_csv(args.articles_to_labels)
    label_names = pd.read_csv(args.label_names)
    cluster_groups = pd.read_csv(args.experiment + "/" + args.cluster_groups)
    article_labels = pd.merge(article_labels, label_names, on='label_id')

    main(args.experiment, article_labels, cluster_groups, args.label_source, args.output_file)
