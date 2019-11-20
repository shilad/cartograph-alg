from beta.cluster_generator import KMeans
from scipy.sparse import csr_matrix
import beta.utils.tfidf_util as utils
import pandas as pd
import numpy as np
import argparse


def create_sparse_label_matrix(article_labels, tf_idf_score):
    """Creates a matrix that contains article ids and label ids,
    the entry of which is the label score from gensim (if available) or tf-idf score."""
    num_row = max(article_labels['article_id']) + 1
    num_col = max(article_labels['label_id']) + 1
    output_matrix = csr_matrix((num_row, num_col), dtype=np.float).toarray()
    # if 'score' not in article_labels.columns:
    article_labels = article_labels.merge(tf_idf_score, on=['article_id', 'label_id'])
    for row in article_labels.itertuples():
        output_matrix[row.article_id][row.label_id] = row.tfidf

    return output_matrix


def main(article_ids, xy_embeddings, articles_to_labels, output_file, label_names, k=9, loss_weight=float(0.2)):
    """
    Generate ${label_types[$i]}_cluster_groups.csv file by joint algorithm minimizing high+low+label loss
    In order to compute label loss, we need an initial tf-idf score after generating original_cluster_groups.csv
    """
    # original cluster
    km = KMeans(k)
    vectors = vanilla_vectors.iloc[:, 1:].values
    orig_groups, orig_average_distance = km.fit_original_kmeans(vectors)
    orig_groups = article_ids.join(pd.DataFrame(orig_groups))
    orig_groups.columns = ['article_id', 'country']
    orig_groups.to_csv('%s/orig_cluster_groups.csv' % (experiment_directory,), index=False)
    articles_to_labels = pd.merge(articles_to_labels, label_names, on="label_id")

    tf_idf_score = utils.calc_tfidf(articles_to_labels, orig_groups, ['article_id', 'label_id', 'tfidf'])

    # joint cluster
    sparse_matrix = create_sparse_label_matrix(articles_to_labels, tf_idf_score)  # #article * #labels wide matrix
    filtered_matrix = sparse_matrix[article_ids['article_id'].values]   # only valid articles to cluster

    joint_alg_groups, joint_average_distance = km.fit_joint_all(vectors, orig_groups, article_ids, xy_embeddings, sparse_matrix, filtered_matrix, loss_weight)
    # joint_alg_groups = article_ids.join(pd.DataFrame(joint_alg_groups))
    joint_alg_groups = pd.DataFrame(joint_alg_groups)
    joint_alg_groups.columns = ['article_id', 'country']
    joint_alg_groups.to_csv(output_file, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster & Label articles in the high dimensional space using K-means')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)  # vanilla vector
    parser.add_argument('--xy_embeddings', required=True) # xy_embedding after umap
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--k', required=True, type=int)

    args = parser.parse_args()

    experiment_directory = args.experiment
    vanilla_vectors = pd.read_csv(args.vectors)
    article_ids = pd.DataFrame(vanilla_vectors['article_id'])
    xy_embeddings = pd.read_csv(args.xy_embeddings)
    articles_to_labels = pd.read_csv(args.articles_to_labels)
    label_names = pd.read_csv(args.label_names)

    main(article_ids, xy_embeddings, articles_to_labels, args.output_file, label_names, args.k)




