from ast import literal_eval
import numpy as np
import pandas as pd
import warnings
import cartograph.label_selector as ls
from scipy.sparse import csr_matrix
import argparse
from sklearn.metrics.pairwise import cosine_distances
warnings.simplefilter(action='ignore', category=FutureWarning)


class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}


    def fit(self, data, ids):
        print("original")
        N, D = data.shape
        K = self.k

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids)
            assert high_dim_dist.shape == (N, K)

            dis_mat = high_dim_dist
            best_group = np.argmin(dis_mat, axis=1)
            dis_mat_df = pd.DataFrame(dis_mat)
            dis_mat_df = ids.join(dis_mat_df)
            # print(dis_mat_df[dis_mat_df['article_id'] == 3103])
            print(dis_mat_df[dis_mat_df['article_id'] == 1994].idxmin(axis=1).iloc[0])
            assert best_group.shape == (N,)

            points_per_group = np.zeros(K) + 1e-6
            np.add.at(points_per_group, best_group, 1)
            # print(points_per_group)

            new_centroids = np.zeros((K, D))
            np.add.at(new_centroids, best_group, data)
            new_centroids /= points_per_group.repeat(D).reshape(K, D)
            # print(new_centroids)

            centroid_changes = np.sum(np.abs(new_centroids - centroids), axis=1)
            assert centroid_changes.shape == (K, )
            max_centroid_change = np.max(centroid_changes)
            # print('change is', max_centroid_change)

            centroids = new_centroids

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break
        # print('points per group before fit with y', points_per_group)
        return best_group

    def fit_with_y(self, data, article_keywords, country_names, ids, k, weight):
        # print(article_and_group.article_id)
        # print("\n----using local neighbor scores")
        print("with y ")

        N, D = data.shape
        K = self.k

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids)  # get cosine distance betw each point and the cenroids, N x k
            assert high_dim_dist.shape == (N, K)
            label_scores = label_affinity(article_keywords, country_names, ids, k)  # get the homogeneity vector, 4097 *
            # print('Label Scores', label_scores[2596])
            # if(i==1):
            #     print(high_dim_dist)
            dis_mat = high_dim_dist * (1 - weight) - label_scores * weight  # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
            dis_mat_df = pd.DataFrame(dis_mat)
            dis_mat_df = ids.join(dis_mat_df)

            # print(label_scores[2596])
            print(dis_mat_df[dis_mat_df['article_id'] == 1994].idxmin(axis=1).iloc[0])
            assert best_group.shape == (N,)

            points_per_group = np.zeros(K) + 1e-6
            np.add.at(points_per_group, best_group, 1)
            # print(points_per_group)

            new_centroids = np.zeros((K, D))
            np.add.at(new_centroids, best_group, data)
            new_centroids /= points_per_group.repeat(D).reshape(K, D)
            # print(new_centroids)

            centroid_changes = np.sum(np.abs(new_centroids - centroids), axis=1)
            assert centroid_changes.shape == (K, )
            max_centroid_change = np.max(centroid_changes)
            # print('change is', max_centroid_change)

            centroids = new_centroids

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break
        # print('points per group AFTER fit with y', points_per_group)
        return best_group

def create_label_matrix(label_matrix):
    """Creates a matrix that contains a article ids and label ids."""
    nrow = max(label_matrix['article_id']) + 1
    ncol = max(label_matrix['label_id']) + 1
    output_matrix = csr_matrix((nrow, ncol), dtype=np.float).toarray()
    for row in label_matrix.itertuples():
        output_matrix[row.article_id][row.label_id] = row.score
    output_matrix = pd.DataFrame(output_matrix)
    output_matrix.index.name = 'article_id'
    return output_matrix


def label_affinity(keyword_scores, country_names, article_ids, k):
    article_label_scores = pd.read_csv(keyword_scores)
    sparse_article_label_scores = create_label_matrix(article_label_scores)
    labeled_clusters = pd.read_csv(country_names)
    # print(labeled_clusters)
    # label_ids = labeled_clusters['label_id'].values
    label_ids = labeled_clusters['label_id']
    # print("_________________")
    # print(list(literal_eval(label_ids[0])))
    # print("_________________")

    label_ids = np.vstack(np.array(list(literal_eval(label_ids[i]))) for i in range(8))
    label_ids = pd.DataFrame(label_ids)
    filtered_sparse = pd.DataFrame()
    for i in range(8):
        xixi = label_ids.loc[i]
        hh = sparse_article_label_scores[xixi]
        filtered_sparse[str(i)] = hh.sum(axis=1)

    # print(filtered_sparse)

    # filtered_sparse = sparse_article_label_scores[label_ids]
    # some article do not have keywords according to article_keywords, therefore we will
    merged_cluster_label_score = pd.merge(ids, filtered_sparse, on='article_id')
    extra_zeros = np.zeros((article_ids.shape[0] - merged_cluster_label_score.shape[0], k))
    # print(merged_cluster_label_score.shape, extra_zeros.shape)
    assert extra_zeros.shape[1] == merged_cluster_label_score.shape[1]-1
    score_dist_mat = np.vstack((merged_cluster_label_score.values[:, 1:], extra_zeros))
    return score_dist_mat





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster & Label articles in the high dimensional space using K-means')
    parser.add_argument('--experiment_directory', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--weight', default=0.1, type=float)
    # arguments for label_selector
    parser.add_argument('--articles_to_labels', required=True)
    parser.add_argument('--label_names', required=True)
    parser.add_argument('--percentile', required=True)
    parser.add_argument('--label_score', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--output_file', required=True)
    # arguments for Label Scoring
    parser.add_argument('--article_keywords', required=True)
    parser.add_argument('--country_names', required=True)
    parser.add_argument('--num_candidates', required=True, type=int)

    args = parser.parse_args()

    # Initial Clustering
    article_vectors = pd.read_csv(args.vectors)
    ids = pd.DataFrame(article_vectors['article_id'])
    X = article_vectors.iloc[:, 1:].values
    km = K_Means(int(args.k))
    init_groups = km.fit(X, ids)
    init_groups = ids.join(pd.DataFrame(init_groups))
    init_groups.columns = ['article_id', 'country']
    init_groups.to_csv('%s/original_cluster_groups.csv' % (args.experiment_directory, ), index=False)

    # Initial Labeling
    article_labels = pd.read_csv(args.articles_to_labels)
    country_clusters = pd.read_csv(args.experiment_directory + args.cluster_groups)
    label_names = pd.read_csv(args.label_names)
    article_labels_orig = pd.merge(article_labels, country_clusters, on='article_id')
    article_labels_orig = pd.merge(article_labels_orig, label_names, on='label_id')
    ls.main(args.experiment_directory, article_labels_orig, args.percentile, args.label_score, "/original_country_labels_no_set.csv", False, args.num_candidates)

    ls.main(args.experiment_directory, article_labels_orig, args.percentile, args.label_score, args.output_file, True, args.num_candidates)


    # Combined Clustering & Labeling
    joint_fit_groups = km.fit_with_y(X, args.article_keywords, args.country_names, ids, args.k, args.weight)
    joint_fit_groups = pd.DataFrame(joint_fit_groups)
    joint_fit_groups = ids.join(joint_fit_groups)
    joint_fit_groups.columns = ['article_id', 'country']

    joint_fit_groups.to_csv('%s/cluster_groups.csv' % (args.experiment_directory, ), index=False)
    article_labels_new = pd.merge(article_labels, joint_fit_groups, on='article_id')
    article_labels_new = pd.merge(article_labels_new, label_names, on='label_id')
    ls.main(args.experiment_directory, article_labels_new, args.percentile, args.label_score, '/new_country_labels.csv', False, args.num_candidates)
    # print(str(json.dumps({'weight': args.weight})))






