"""

A modified Kmeans clustering algorithm that optimizes clustering, labeling,
as well as low dimensional embeddings.

Authors: Jonathan Scott, Lu Li

"""
from ast import literal_eval
import numpy as np
import random
import pandas as pd
import warnings
import cartograph.label_selector as ls
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from pandas._libs import json
warnings.simplefilter(action='ignore', category=FutureWarning)


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
    output_matrix = pd.DataFrame(output_matrix)
    output_matrix.index.name = 'article_id'
    # output_matrix = output_matrix.rename(columns= lambda x: "label_id_" + str(x) )
    # print(output_matrix)

    return output_matrix

def get_label_score_new_matrix(article_labels, tf_idf_score, best_group, ids):
    ids_with_group = ids.join(pd.DataFrame(best_group))
    ids_with_group = pd.DataFrame(ids_with_group)
    ids_with_group.columns = ['article_id', 'country']
    country_articles = ids_with_group.groupby('country')['article_id'].apply(list)
    sparse = create_sparse_label_matrix(article_labels, tf_idf_score)
    sparse2 = sparse.iloc[ids['article_id'].values.tolist()]
    top_labels_list = []
    top_scores_list = []
    for i in range(country_articles.shape[0]):
        scores = sparse.iloc[country_articles[i]].mean(axis=0)
        scores_df = pd.DataFrame(scores)
        scores_df.columns = ['label_score']
        top = scores_df.sort_values(by=['label_score'], ascending=False)[0:20]
        top_scores_list.append(top['label_score'].values.tolist())
        top_labels_list.append(top.index.values.tolist())
    filtered = sparse2.iloc[:, top_labels_list[0]]
    # print(filtered.shape)
    distance = cosine_distances(filtered, top_scores_list)
    distance = pd.DataFrame(distance)
    print(distance.shape)
    return distance
def get_label_score_matrix(article_labels, country_labels, article_ids, k, tf_idf_score):
    """
    :param country_labels: a data frame of label candidate sets and the corresponding sets of label ids.
    :return: a matrix that contains the label score between each article and each candidate label set
    """
    n = len(article_ids)

    article_labels = pd.read_csv(article_labels)
    country_labels = pd.read_csv(country_labels)

    sparse_label_matrix = create_sparse_label_matrix(article_labels, tf_idf_score)

    # the matrix of only the articles
    article_country_score_matrix = pd.DataFrame()
    for country in range(k):
        # Get label ids for country
        candidate_label_ids = country_labels[country_labels.country == country].label_id.values
        # get the entire columns of label scores given a candidate label id set
        label_score_columns = sparse_label_matrix[candidate_label_ids]
        # sum up label scores between each article and each label candidate set
        label_score_sum = label_score_columns.sum(axis=1)
        article_country_score_matrix[str(country)] = label_score_sum  # append the label score as a new column

    merged_cluster_label_score = pd.merge(ids, article_country_score_matrix, on='article_id')
    merged_cluster_label_score = merged_cluster_label_score.values[:, 1:]

    # some articles do not have keyword label scores, the code below is to make up the missing rows
    # by appending rows of zeros
    if n != merged_cluster_label_score.shape[0]:
        extra_zeros = np.zeros((article_ids.shape[0] - merged_cluster_label_score.shape[0], k))
        assert extra_zeros.shape[1] == merged_cluster_label_score.shape[1] - 1
        label_score_mat = np.vstack((merged_cluster_label_score.values[:, 1:], extra_zeros))
    else:
        label_score_mat = merged_cluster_label_score
    assert label_score_mat.shape == (n, k)
    return label_score_mat


def get_vector_centroid_distance(article_vectors, centroids, membership):
    """
    Returns a list of high dimensional distance between each point and its cluster's centroid
    """
    distances = cosine_distances(article_vectors, centroids)
    distance = []
    for i in range(article_vectors.shape[0]):
        dist = np.abs(distances[i, membership[i]])
        distance.append(dist)
    return distance


def get_final_labels(label_scores, final_groups, country_labels, k, tf_idf_score):
    """Returns the labels with the highest sum of scores based on the articles within each cluster."""
    article_label_scores = pd.read_csv(label_scores)
    country_labels = pd.read_csv(country_labels)

    sparse_label_scores = create_sparse_label_matrix(article_label_scores, tf_idf_score)
    lists_of_articles_by_country = final_groups.groupby('country')['article_id'].apply(list)
    articles_by_country_df = pd.DataFrame(lists_of_articles_by_country)
    country_ids = country_labels['country']

    score_labels_ids = []
    for country in range(k):
        # Get label ids for country
        label_id = country_labels[country_labels.country == country].label_id.values
        label_score_column = sparse_label_scores[label_id]
        country_members = articles_by_country_df['article_id'].iloc[country]
        potential_labels = label_score_column.ix[country_members].sum(axis=0)
        score_labels_ids.append(potential_labels.argmax())

    score_labels_df = pd.DataFrame(score_labels_ids)
    labels = []

    for i in range(len(score_labels_ids)):
        labels.append(label_names[label_names['label_id'] == score_labels_ids[i]].label.values[0])
    # print(labels)
    score_labels_df['label_name'] = labels
    country_ids = pd.DataFrame(country_ids)
    score_labels_df = country_ids.join(score_labels_df)
    score_labels_df.columns = ['country', 'label_id', 'label_name']
    return score_labels_df


def get_centroid_distance(data, centroids, membership):
    """ Calculate the mean high dimensional distances between each point and its cluster's centroid """
    distances = cosine_distances(data, centroids)
    total_distance = 0
    for article in range(data.shape[0]):
        dist = np.abs(distances[article, membership[article]])
        total_distance += dist
    mean_distance = total_distance / data.shape[0]
    return mean_distance


class KMeans:
    def __init__(self, k=7, tolerance=0.00001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def fit_original_kmeans(self, data):
        """
        :param data: The original article vectors.
        :return: cluster groups fit using the original Kmeans clustering algorithm.
        """
        N, D = data.shape  # number of data points, dimensionality of each vector
        K = self.k  # number of clusters

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids)
            assert high_dim_dist.shape == (N, K)

            dis_mat = high_dim_dist
            best_group = np.argmin(dis_mat, axis=1)
            assert best_group.shape == (N,)

            points_per_group = np.zeros(K) + 1e-6  # + 1e-6 to avoid dividing by zero error
            np.add.at(points_per_group, best_group, 1)

            new_centroids = np.zeros((K, D))
            np.add.at(new_centroids, best_group, data)
            new_centroids /= points_per_group.repeat(D).reshape(K, D)

            centroid_changes = np.sum(np.abs(new_centroids - centroids), axis=1)
            assert centroid_changes.shape == (K, )
            max_centroid_change = np.max(centroid_changes)

            centroids = new_centroids

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break
        # get the average distance from an article to its centroid
        average_distance = get_centroid_distance(data, centroids, best_group)
        # a list of distances from each point to its cluster centroid, to filter articles later used to label
        distance = get_vector_centroid_distance(data, centroids, best_group)
        return best_group, distance, average_distance

    def fit_joint_all(self, data, article_keywords, country_labels, ids, k, weight, embeddings, tf_idf_score):
        N, D = data.shape
        K = self.k
        embeddings = embeddings.iloc[:, 1:].values


        # initialize the first 'k' elements in the dataset to be the initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)
        centroids2 = np.stack(embeddings[:K])   # low dimensional clustering
        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids)  # get cosine distance betw each point and the cenroids, N x k
            assert high_dim_dist.shape == (N, K)

            # Calculate normalized euclidean distance

            low_dim_dist = euclidean_distances(embeddings, centroids2)
            xy_range = (np.max(embeddings) - np.min(embeddings))
            max_dist = np.sqrt(xy_range * xy_range + xy_range * xy_range)
            low_dim_dist /= max_dist
            if(i == 0):
                res = []

                for j in range(data.shape[0]):
                    res.append(random.randint(0, 6))

                best_group = res
            # else: best_group = best_group
            dis_mat = high_dim_dist * (0.95 - weight) + low_dim_dist * 0.05
            print(dis_mat)
            best_group = np.argmin(dis_mat, axis=1)

            label_scores = get_label_score_new_matrix(article_labels, tf_idf_score, best_group, ids)
            dis_mat_2 = dis_mat - label_scores * weight
            print(dis_mat_2)
            best_group = np.argmin(dis_mat_2.values.tolist(), axis=1)
            assert best_group.shape == (N,)

            points_per_group = np.zeros(K) + 1e-6
            np.add.at(points_per_group, best_group, 1)

            new_centroids = np.zeros((K, D))
            np.add.at(new_centroids, best_group, data)
            new_centroids /= points_per_group.repeat(D).reshape(K, D)

            centroid_changes = np.sum(np.abs(new_centroids - centroids), axis=1)
            assert centroid_changes.shape == (K, )
            max_centroid_change = np.max(centroid_changes)

            new_centroids2 = np.zeros((K, 2))
            np.add.at(new_centroids2, best_group, embeddings)
            new_centroids2 /= points_per_group.repeat(2).reshape(K, 2)

            centroids = new_centroids

            centroids2 = new_centroids2
            ave_distance = get_centroid_distance(data, centroids, best_group)
            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break
        distance = get_vector_centroid_distance(data, centroids, best_group)
        return best_group, distance, ave_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster & Label articles in the high dimensional space using K-means')
    parser.add_argument('--experiment_directory', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--weight', default=0.1, type=float)
    parser.add_argument('--xy_embeddings', required=True)

    # arguments for label_selector
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--percentile', required=True)
    parser.add_argument('--label_score', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--output_file', required=True)

    # arguments for Label Scoring
    parser.add_argument('--article_keywords', required=True)
    parser.add_argument('--country_labels', required=True)
    parser.add_argument('--num_candidates', required=True, type=int)
    parser.add_argument('--tf_idf_score_file', required=True)

    args = parser.parse_args()
    xy_embeddings = pd.read_csv(args.xy_embeddings)

    # Original Clustering
    article_vectors = pd.read_csv(args.vectors)
    ids = pd.DataFrame(article_vectors['article_id'])
    X = article_vectors.iloc[:, 1:].values
    km = KMeans(args.k)
    orig_groups, orig_distance_list, orig_average_distance = km.fit_original_kmeans(X)
    orig_groups = ids.join(pd.DataFrame(orig_groups))
    orig_groups.columns = ['article_id', 'country']
    orig_groups['distance'] = orig_distance_list
    orig_groups.to_csv('%s/original_cluster_groups.csv' % (args.experiment_directory,), index=False)

    # Original Labeling
    article_labels = pd.read_csv(args.articles_to_labels)
    label_names = pd.read_csv(args.label_names)
    article_labels_orig = pd.merge(article_labels, orig_groups, on='article_id')
    article_labels_orig = pd.merge(article_labels_orig, label_names, on='label_id')

    ls.main(args.experiment_directory, article_labels_orig, args.percentile, args.label_score, args.experiment_directory + '/original_country_labels.csv', args.num_candidates)

    # Joint Clustering
    tf_idf_score = pd.read_csv(args.experiment_directory + args.tf_idf_score_file)
    joint_alg_groups, joint_distance_list, joint_average_distance = km.fit_joint_all(X, args.article_keywords, args.experiment_directory + '/top_labels.csv', ids, args.k, args.weight, xy_embeddings, tf_idf_score)
    joint_alg_groups = pd.DataFrame(joint_alg_groups)
    joint_alg_groups = ids.join(joint_alg_groups)
    joint_alg_groups.columns = ['article_id', 'country']
    joint_alg_groups['distance'] = joint_distance_list
    joint_alg_groups.to_csv('%s/new_cluster_groups.csv' % (args.experiment_directory,), index=False)
    joint_alg_groups.to_csv('%s/score_cluster_groups.csv' % (args.experiment_directory,), index=False)

    # Joint Labeling
    article_labels_new = pd.merge(article_labels, joint_alg_groups, on='article_id')
    article_labels_new = pd.merge(article_labels_new, label_names, on='label_id')
    ls.main(args.experiment_directory, article_labels_new, args.percentile, args.label_score, args.experiment_directory + '/new_country_labels.csv',args.num_candidates)

    # get labels based on label scores instead of running tfidf again
    score_based_labels = get_final_labels(args.article_keywords, joint_alg_groups, args.experiment_directory + '/top_labels.csv', args.k, tf_idf_score)
    score_based_labels.to_csv(args.experiment_directory + "/score_country_labels.csv", index=True)
    #
    # print(str(json.dumps(orig_average_distance)))
    # print(str(json.dumps(joint_average_distance)))
    #





