import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances


def get_mean_centroid_distance(data, centroids, membership):
    """
    Calculate the mean high dimensional distances between each point and its cluster's centroid
    """
    distances = cosine_distances(data, centroids)
    total_distance = 0
    for article in range(data.shape[0]):
        dist = np.abs(distances[article, membership[article]])
        total_distance += dist
    mean_distance = total_distance / data.shape[0]
    return mean_distance

def generate_country_matrix(groups):
    """
    Creates a matrix that contains article ids and label ids,
    the entry of which is the label score from gensim (if available) or tf-idf score.
    """
    num_row = max(groups.iloc[:,1]) + 1
    num_col = max(groups.iloc[:,0]) + 1
    output_matrix = csr_matrix((num_row, num_col), dtype=np.float).toarray()
    for row in groups.itertuples():
        output_matrix[row.country][row.article_id] = 1
    return output_matrix

def assign_best_groups(dist_mat, article_ids):
    dis_mat = pd.DataFrame(dist_mat)
    best_group = dis_mat.idxmin(axis=1)
    best_group = pd.DataFrame(best_group, columns=["country"])
    best_group = article_ids.join(best_group)
    return pd.DataFrame(best_group)


class KMeans:
    def __init__(self, k=7, tolerance=0.00005, max_iterations=500):
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
        average_distance = get_mean_centroid_distance(data, centroids, best_group)
        return best_group, average_distance

    def fit_joint_all(self, vectors, orig_groups, article_ids, xy_embeddings, sparse_matrix, filtered_matrix, loss_weight):
        N, D = vectors.shape
        K = self.k
        embeddings = xy_embeddings.iloc[:, 1:].values
        best_group = orig_groups

        # initialize the first 'k' elements in the dataset to be the initial centroids
        high_centroid = np.stack(vectors[:K])
        assert high_centroid.shape == (K, D)
        low_centroid = np.stack(embeddings[:K])   # low dimensional clustering

        for i in range(self.max_iterations):
            high_dim_dist = cosine_distances(vectors, high_centroid)   # get cosine distance betw each point and the cenroids, N x k
            assert high_dim_dist.shape == (N, K)

            # Calculate normalized euclidean distance
            low_dim_dist = euclidean_distances(embeddings, low_centroid)
            xy_range = (np.max(embeddings) - np.min(embeddings))
            max_dist = np.sqrt(xy_range * xy_range + xy_range * xy_range)
            low_dim_dist /= max_dist

            country_matrix = generate_country_matrix(best_group)
            country_label = country_matrix.dot(sparse_matrix) #Todo: fixed num articles
            #Todo: Ask Shilad
            # country_label = normalize(country_label, axis=1)
            label_scores = cosine_distances(filtered_matrix, country_label)

            # Calculate loss
            dis_mat = high_dim_dist * (0.95 - loss_weight) + low_dim_dist * 0.05 + label_scores * loss_weight
            best_group = assign_best_groups(dis_mat, article_ids)
            assert best_group.shape == (N, 2)

            # calculate the # of articles per group
            points_per_group = np.zeros(K) + 1e-6
            np.add.at(points_per_group, best_group['country'], 1)

            # calculate the new centroid by averaging the new centroid at each cluster in both high and low dim
            high_centroid_new = np.zeros((K, D))
            np.add.at(high_centroid_new, best_group['country'], vectors)
            high_centroid_new /= points_per_group.repeat(D).reshape(K, D)

            low_centroid_new = np.zeros((K, 2))
            np.add.at(low_centroid_new, best_group['country'], embeddings)
            low_centroid_new /= points_per_group.repeat(2).reshape(K, 2)

            high_centroid = high_centroid_new
            low_centroid = low_centroid_new

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            centroid_changes = np.sum(np.abs(high_centroid_new - high_centroid), axis=1)
            assert centroid_changes.shape == (K,)
            max_centroid_change = np.max(centroid_changes)
            if max_centroid_change < self.tolerance:
                break
        mean_distance = get_mean_centroid_distance(vectors, high_centroid, best_group['country'])
        return best_group, mean_distance
