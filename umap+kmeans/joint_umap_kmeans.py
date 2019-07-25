"""
Joint algorithm which alternates between iterations of Kmeans clustering and UMAP embedding algorithms.

Authors: Jonathan Scott, Shilad Sen, Lu Li
"""
import sys
import numpy as np
import pandas as pd
from pandas._libs import json
from sklearn.neighbors import NearestNeighbors
from cartograph.xy_embed.umap_embed import create_embeddings
import argparse
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


class K_Means:
    def __init__(self, k=3, tolerance=0.000001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}


    def fit(self, article_vectors):
        data = article_vectors.iloc[:, 1:].values
        N, D = data.shape
        K = self.k

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids)  # get cosine distance betw each point and the cenroids, N x k
            assert high_dim_dist.shape == (N, K)
            dis_mat = high_dim_dist   # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
            assert best_group.shape == (N,)

            points_per_group = np.zeros(K) + 1e-6  # get points per group
            np.add.at(points_per_group, best_group, 1)

            new_centroids = np.zeros((K, D))  # new centroids
            np.add.at(new_centroids, best_group, data)
            new_centroids /= points_per_group.repeat(D).reshape(K, D)

            centroid_changes = np.sum(np.abs(new_centroids - centroids), axis=1)
            assert centroid_changes.shape == (K,)
            max_centroid_change = np.max(centroid_changes)

            centroids = new_centroids
            distance = evaluate_clusters(data, centroids, best_group)
            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break
        return best_group, distance

    def fit_local_neighbor(self, article_vectors, embeddings, article_and_group, weight, n_neighbors):

        data = article_vectors.iloc[:, 1:].values
        N, D = data.shape
        K = self.k

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids) # get cosine distance betw each point and the cenroids, N x k
            assert high_dim_dist.shape == (N, K)
            neighbor_score = get_neighbor_score(embeddings, article_and_group.country.values, self.k, n_neighbors)  # get the homogeneity vector, 4097 *

            dis_mat = high_dim_dist * (1.0 - weight) + neighbor_score * weight  # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
            assert best_group.shape == (N,)

            points_per_group = np.zeros(K) + 1e-6
            np.add.at(points_per_group, best_group, 1)

            new_centroids = np.zeros((K, D))
            np.add.at(new_centroids, best_group, data)
            new_centroids /= points_per_group.repeat(D).reshape(K, D)

            centroid_changes = np.sum(np.abs(new_centroids - centroids), axis=1)
            assert centroid_changes.shape == (K, )
            max_centroid_change = np.max(centroid_changes)

            centroids = new_centroids
            distance = evaluate_clusters(data, centroids, best_group)

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break

        return best_group, distance

    def fit_low_dim(self, article_vectors, embeddings, weight):

        data = article_vectors.iloc[:, 1:].values
        embeddings = embeddings.iloc[:, 1:].values
        # print(article_and_group.article_id)
        N, D = data.shape
        K = self.k

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
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

            dis_mat = high_dim_dist * (1.0 - weight) + low_dim_dist * weight  # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
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
            distance = evaluate_clusters(data, centroids, best_group)
            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break

        return best_group, distance


def get_neighbor_score(embedding, groups, k, n_neighbors=20):
    """
    Get the normalized number of clusters that a point's nearest k neighbors belong to.
    :param embedding: original embedding
    :param groups: original cluster groups
    :param k: number of clusters
    :param n_neighbors: number of nearest neighbors
    :return: homogeneity score
    """
    n = embedding.shape[0]
    # Get nearest neighbors and distances
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, algorithm='kd_tree').fit(embedding)
    indices = nbrs.kneighbors(embedding, return_distance=False)[:, 1:]  # drop nearest neighbor, which is the point itself
    assert indices.shape == (n, n_neighbors)
    groups = np.array(groups)
    assert groups.shape == (n,)
    neighbor_groups = groups[indices]
    counts = np.zeros((n, k), dtype=np.float)
    for i in range(k):
        counts[:, i] = np.sum(neighbor_groups == i, axis=1)
    counts /= n_neighbors
    return 1.0 - counts


def evaluate_clusters(data, centroids, membership):
    # Calculate the mean high dimensional distances between each point and its cluster's centroid
    distances = cosine_distances(data, centroids)
    total_distance = 0
    for i in range(data.shape[0]):
        dist = np.abs(distances[i, membership[i]])
        total_distance += dist
    total_distance /= data.shape[0]
    return total_distance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Joint algorithm in terms of UMAP and Kmeans.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--k', default=8, type=int)
    parser.add_argument('--weight', default=0.1, type=float)
    parser.add_argument('--num_neighbors', default=20, type=int)
    args = parser.parse_args()

    if args.weight < 0.0 or args.weight > 1.0:
        sys.stderr.write('Weight must be between 0.0 and 1.0\n')
        sys.exit(1)

    article_vectors = pd.read_csv(args.vectors)
    km = K_Means(args.k)

    orig_groups, orig_total_distance = km.fit(article_vectors)

    ids = pd.DataFrame(article_vectors['article_id'])

    orig_groups = ids.join(pd.DataFrame(orig_groups))
    orig_groups.columns = ['article_id', 'country']
    # init_y = create_embeddings(args.vectors)
    orig_embed = create_embeddings(args.vectors, clusters=orig_groups, tw=args.weight)
    orig_embed.to_csv('%s/original_xy_embeddings.csv' % (args.experiment,), index=False)
    orig_groups.to_csv('%s/original_cluster_groups.csv' % (args.experiment,), index=False)

    low_groups, low_total_distance = km.fit_low_dim(article_vectors, orig_embed, args.weight)
    low_groups = pd.DataFrame(low_groups)
    low_groups = ids.join(low_groups)
    low_groups.columns = ['article_id', 'country']
    low_embed = create_embeddings(args.vectors, clusters=low_groups, tw=args.weight)

    local_groups, local_total_distance = km.fit_local_neighbor(article_vectors, orig_embed, orig_groups, args.weight, args.num_neighbors)
    local_groups = pd.DataFrame(local_groups)
    local_groups = ids.join(local_groups)
    local_groups.columns = ['article_id', 'country']
    local_embed = create_embeddings(args.vectors, clusters=local_groups, tw=args.weight)

    low_groups.to_csv('%s/low_cluster_groups.csv' % (args.experiment,), index=False)
    low_embed.to_csv('%s/low_xy_embeddings.csv' % (args.experiment,), index=False)
    local_embed.to_csv('%s/local_xy_embeddings.csv' % (args.experiment,), index=False)
    local_groups.to_csv('%s/local_cluster_groups.csv' % (args.experiment,), index=False)

    print(str(json.dumps(orig_total_distance)))
    print(str(json.dumps(local_total_distance)))
    print(str(json.dumps(low_total_distance)))


#iterative
    # for i in range(5):
    #     joint_fit_groups = km.fit_with_y2(X, joint_embeddings, args.weight)
    #     joint_fit_groups = pd.DataFrame(joint_fit_groups)
    #     joint_fit_groups = ids.join(joint_fit_groups)
    #     joint_fit_groups.columns = ['article_id', 'country']
    #     joint_embeddings = create_embeddings(args.vectors, clusters=joint_fit_groups, tw=float(args.weight))
    # print("\nWith iterative steps")