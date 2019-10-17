from beta.cluster_generator import KMeans
import beta.tfidf_pmi_generator as tfidf_generator
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np
import argparse
import logging


def create_sparse_label_matrix(article_labels, tf_idf_score):
    """Creates a matrix that contains article ids and label ids,
    the entry of which is the label score from gensim (if available) or tf-idf score."""
    num_row = max(article_labels['article_id']) + 1
    num_col = max(article_labels['label_id']) + 1
    output_matrix = csr_matrix((num_row, num_col), dtype=np.float).toarray()
    if 'score' not in article_labels.columns:
        article_labels = article_labels.merge(tf_idf_score, on=['article_id', 'label_id'])
        for row in article_labels.itertuples():
            output_matrix[row.article_id][row.label_id] = row.tfidf
    else:
        for row in article_labels.itertuples():
            output_matrix[row.article_id][row.label_id] = row.score
    return output_matrix

def main(experiment_directory, article_ids, xy_embeddings,
         articles_to_labels, tf_idf_score_file, loss_weight, output_file):
    # joint cluster
    sparse_matrix = create_sparse_label_matrix(articles_to_labels, tf_idf_score_file)  # #article * #labels wide matrix
    filtered_matrix = sparse_matrix[article_ids['article_id'].values]   # only valid articles to cluster
    joint_alg_groups, joint_average_distance = km.fit_joint_all(vectors, orig_groups, article_ids, xy_embeddings, sparse_matrix, filtered_matrix, loss_weight)
    joint_alg_groups = pd.DataFrame(joint_alg_groups)
    # joint_alg_groups = article_ids.join(joint_alg_groups) # todo: fix!!!!!!!
    joint_alg_groups.columns = ['article_id', 'country']
    joint_alg_groups.to_csv(output_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster & Label articles in the high dimensional space using K-means')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)  # vanilla vector
    parser.add_argument('--xy_embeddings', required=True) # xy_embedding after umap
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--tf_idf_score_file', required=True)
    parser.add_argument('--loss_weight', required=True, type=float)
    parser.add_argument('--output_file', required=True)

    args = parser.parse_args()

    experiment_directory = args.experiment
    vanilla_vectors = pd.read_csv(args.vectors)
    article_ids = pd.DataFrame(vanilla_vectors['article_id'])
    xy_embeddings = pd.read_csv(args.xy_embeddings)
    articles_to_labels = pd.read_csv(args.articles_to_labels)
    loss_weight = args.loss_weight

    km = KMeans()
    # original clustering
    vectors = vanilla_vectors.iloc[:, 1:].values
    orig_groups, orig_average_distance = km.fit_original_kmeans(vectors)
    orig_groups = article_ids.join(pd.DataFrame(orig_groups))
    orig_groups.columns = ['article_id', 'country']
    orig_groups.to_csv('%s/orig_cluster_groups.csv' % (experiment_directory,), index=False)
    tfidf_generator.main(experiment_directory, articles_to_labels, orig_groups)
    tf_idf_score_file = pd.read_csv(args.tf_idf_score_file)

    main(experiment_directory, article_ids, xy_embeddings, articles_to_labels, tf_idf_score_file, loss_weight, args.output_file)




