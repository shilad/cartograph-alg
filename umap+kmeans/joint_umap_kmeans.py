import numpy as np
import pandas as pd
from pandas._libs import json
from scipy import spatial
from sklearn.neighbors import NearestNeighbors
from cartograph.xy_embed.umap_embed import create_embeddings
import argparse
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def fit(self, data):
        self.centroids = {}
        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        for i in range(self.k):
            self.centroids[i] = data[i]
        # begin iterations
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []
            # find the distance between the point and cluster; choose the nearest centroid
            for features in data:
                distances = [spatial.distance.cosine(features, self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)
            previous = dict(self.centroids)
            # average the cluster datapoints to re-calculate the centroids
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)
            isOptimal = True

            for centroid in self.centroids:

                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if isOptimal:
                break
        classes = []
        for feature in data:
            distances = [spatial.distance.cosine(feature, self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))
            classes.append(classification)
        return classes

    def fit_with_y(self, data, embeddings, article_and_group, weight, n_neighbors):
        # print(article_and_group.article_id)
        # print("\n----using local neighbor scores")

        N, D = data.shape
        K = self.k

        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(data, centroids)  # get cosine distance betw each point and the cenroids, N x k
            assert high_dim_dist.shape == (N, K)
            neighbor_score = kmeans_modified(embeddings, article_and_group.country.values, self.k, n_neighbors)  # get the homogeneity vector, 4097 *
            # print(neighbor_score)

            dis_mat = high_dim_dist + neighbor_score * float(weight) # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
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

        return best_group

    def fit_with_y2(self, data, embeddings, weight):
        # print("\n----using low dimensional distances")
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

            # print(high_dim_dist.mean(), low_dim_dist.mean())

            dis_mat = high_dim_dist + low_dim_dist * float(weight)  # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
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

            new_centroids2 = np.zeros((K, 2))
            np.add.at(new_centroids2, best_group, embeddings)
            new_centroids2 /= points_per_group.repeat(2).reshape(K, 2)

            centroids = new_centroids
            centroids2 = new_centroids2

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break

        return best_group

    def fit_with_y3(self, data, embeddings, article_and_group, weight_high, weight_low, n_neighbors):
        print("\n----using combined distances")
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

            # print(high_dim_dist.mean(), low_dim_dist.mean())
            neighbor_score = kmeans_modified(embeddings, article_and_group.country.values, self.k, n_neighbors)  # get the homogeneity vector, 4097 *

            dis_mat = high_dim_dist * float(weight_high) + (neighbor_score + low_dim_dist) * float(weight_low) # get the distance matrix, 4097 * 8
            best_group = np.argmin(dis_mat, axis=1)
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

            new_centroids2 = np.zeros((K, 2))
            np.add.at(new_centroids2, best_group, embeddings)
            new_centroids2 /= points_per_group.repeat(2).reshape(K, 2)

            centroids = new_centroids
            centroids2 = new_centroids2

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            if max_centroid_change < self.tolerance:
                break

        return best_group


def kmeans_modified(embedding, groups, k, n_neighbors=20):
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
        counts[:,i] = np.sum(neighbor_groups == i, axis=1)
    counts /= n_neighbors

    return 1.0 - counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster articles in the high dimensional space using K-means or '
                                                 'hdbscan.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--k', default=8)
    parser.add_argument('--weight', default=0.1)
    parser.add_argument('--num_neighbors', default=20)

    args = parser.parse_args()

    article_vectors = pd.read_csv(args.vectors)
    X = article_vectors.iloc[:, 1:].values
    km = K_Means(int(args.k))

    init_groups = km.fit(X)

    ids = pd.DataFrame(article_vectors['article_id'])
    init_groups = ids.join(pd.DataFrame(init_groups))
    init_groups.columns = ['article_id', 'country']
    # init_y = create_embeddings(args.vectors)
    init_y = create_embeddings(args.vectors, clusters=init_groups, tw=float(args.weight))

    init_y.to_csv('%s/original_xy_embeddings.csv' % (args.experiment, ), index=False)
    init_groups.to_csv('%s/original_cluster_groups.csv' % (args.experiment, ), index=False)

    # joint_fit_groups = km.fit(X)
    # joint_fit_groups = km.fit_with_y3(X, init_y, init_groups, args.weight_high, args.weight_low, int(args.num_neighbors))

    joint_fit_groups = km.fit_with_y2(X, init_y, args.weight)
    joint_fit_groups = pd.DataFrame(joint_fit_groups)
    joint_fit_groups = ids.join(joint_fit_groups)
    joint_fit_groups.columns = ['article_id', 'country']

    # tw = float(float(args.weight_low)/float(args.weight_high))
    # tw = 0.05
    joint_embeddings = create_embeddings(args.vectors, clusters=joint_fit_groups, tw=float(args.weight))

    # joint_embeddings.to_csv('%s/xy_embeddings.csv' % (args.experiment, ), index=False)
    # joint_fit_groups.to_csv('%s/cluster_groups.csv' % (args.experiment, ), index=False)
    print(str(json.dumps({'weight': args.weight})))

    joint_fit_groups_y = km.fit_with_y(X, joint_embeddings, joint_fit_groups, args.weight, int(args.num_neighbors))
    joint_fit_groups_y = pd.DataFrame(joint_fit_groups_y)
    joint_fit_groups_y = ids.join(joint_fit_groups_y)
    joint_fit_groups_y.columns = ['article_id', 'country']
    joint_embeddings_y = create_embeddings(args.vectors, clusters=joint_fit_groups_y, tw=float(args.weight))


    # for i in range(5):
    #     joint_fit_groups = km.fit_with_y2(X, joint_embeddings, args.weight)
    #     joint_fit_groups = pd.DataFrame(joint_fit_groups)
    #     joint_fit_groups = ids.join(joint_fit_groups)
    #     joint_fit_groups.columns = ['article_id', 'country']
    #     joint_embeddings = create_embeddings(args.vectors, clusters=joint_fit_groups, tw=float(args.weight))
    # print("\nWith iterative steps")


    joint_embeddings.to_csv('%s/y_2_xy_embeddings.csv' % (args.experiment, ), index=False)
    joint_embeddings_y.to_csv('%s/y_xy_embeddings.csv' % (args.experiment, ), index=False)
    joint_fit_groups.to_csv('%s/y_2_cluster_groups.csv' % (args.experiment, ), index=False)
    joint_fit_groups_y.to_csv('%s/y_cluster_groups.csv' % (args.experiment, ), index=False)







