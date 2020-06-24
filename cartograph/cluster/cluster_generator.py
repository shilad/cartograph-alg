import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from sklearn.preprocessing import normalize
import umap


def get_mean_centroid_distance(data, centroids, membership, dimenstion="high"):
    """
    Calculate the mean high dimensional distances between each point and its cluster's centroid
    """
    if dimenstion is "high":
        distances = cosine_distances(data, centroids)
    else:
        distances = euclidean_distances(data, centroids)
    total_distance = 0
    for article in range(data.shape[0]):
        dist = np.abs(distances[article, membership[article]])
        total_distance += dist
    mean_distance = total_distance / data.shape[0]
    return mean_distance


def generate_country_matrix(groups, article_ids):
    """
    Creates a matrix that contains article ids and label ids,
    the entry of which is 1 is an article belongs to a country.
    """
    num_row = max(groups.iloc[:,1]) + 1
    num_col = max(groups.iloc[:,0]) + 1
    output_matrix = csr_matrix((num_row, num_col), dtype=np.float).toarray()
    for row in groups.itertuples():
        output_matrix[row.country][row.article_id] = 1
    output_matrix = output_matrix[:, article_ids['article_id'].values]
    return output_matrix


def assign_best_groups(dist_mat, article_ids):
    dis_mat = pd.DataFrame(dist_mat)
    best_group = dis_mat.idxmin(axis=1)
    best_group = pd.DataFrame(best_group, columns=["country"])
    best_group = article_ids.join(best_group)
    return pd.DataFrame(best_group)


class KMeans:
    def __init__(self, k=9, tolerance=0.00005, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def fit_original_kmeans(self, data, iter = 500):
        """
        :param data: The original article vectors.
        :return: cluster groups fit using the original Kmeans clustering algorithm.
        """
        data = data.iloc[:, 1:].values
        N, D = data.shape  # number of data points, dimensionality of each vector
        K = self.k  # number of clusters
        # initialize the centroids, the first 'k' elements in the dataset will be our initial centroids
        centroids = np.stack(data[:K])
        assert centroids.shape == (K, D)

        # begin iterations
        for i in range(iter):
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
        return best_group, average_distance, centroids

    def fit_joint_all(self, vectors, orig_groups, article_ids, xy_embeddings, filtered_label_matrix, loss_weight, low_weight, output_embedding):
        data = vectors.iloc[:, 1:].values
        N, D = data.shape
        K = self.k
        best_group = orig_groups

        # initialize the first 'k' elements in the dataset to be the initial centroids
        (high_centroid, dist, centroids) = self.fit_original_kmeans(vectors, 3)

        # random embeddings
        max_val, min_val = xy_embeddings[["x", "y"]].max(axis=0), xy_embeddings[["x", "y"]].min(axis=0)
        max_x, max_y = max_val['x'], max_val['y']
        min_x, min_y = min_val['x'], min_val['y']
        random_x = np.random.uniform(min_x, max_y, N)
        random_y = np.random.uniform(min_y, max_y, N)
        embeddings = np.column_stack((random_x, random_y))

        high_centroid = np.stack(centroids[:K])

        assert high_centroid.shape == (K, D)
        low_centroid = np.stack(embeddings[:K])   # low dimensional clustering
        points = "spectral"
        iterations = 0
        for i in range(self.max_iterations):
            iterations += 1
            # get cosine distance betw each point and the cenroids, N x k
            high_dim_dist = cosine_distances(data, high_centroid)
            assert high_dim_dist.shape == (N, K)

            # Calculate normalized euclidean distance in low dimensional space
            low_dim_dist = euclidean_distances(embeddings, low_centroid)
            xy_range = (np.max(embeddings) - np.min(embeddings))
            max_dist = np.sqrt(xy_range * xy_range + xy_range * xy_range)
            low_dim_dist /= max_dist

            # Calculate the label distance
            country_matrix = generate_country_matrix(best_group, article_ids)
            country_label = country_matrix.dot(filtered_label_matrix)
            country_label = normalize(country_label, axis=1)
            label_dist = cosine_distances(filtered_label_matrix, country_label)

            # Calculate loss
            dis_mat = high_dim_dist * round(1 - low_weight - loss_weight, ndigits=2) + low_dim_dist * low_weight + label_dist * loss_weight
            best_group = assign_best_groups(dis_mat, article_ids)
            assert best_group.shape == (N, 2)

            # calculate the # of articles per group
            points_per_group = np.zeros(K) + 1e-6
            np.add.at(points_per_group, best_group['country'], 1)

            # calculate the new high dim centroid by averaging the new centroid at each cluster in both high and low dim
            high_centroid_new = np.zeros((K, D))
            np.add.at(high_centroid_new, best_group['country'], data)
            high_centroid_new /= points_per_group.repeat(D).reshape(K, D)

            low_centroid_new = np.zeros((K, 2))
            np.add.at(low_centroid_new, best_group['country'], embeddings)
            low_centroid_new /= points_per_group.repeat(2).reshape(K, 2)

            # perform umap again and feed in the new best group as the cluster information
            # df = pd.merge(vectors, best_group, on='article_id')

            points = umap.UMAP(metric='cosine', spread=1.0, n_epochs=200, init=points).fit_transform(
                vectors.iloc[:, 1:], y=best_group['country'])
            embeddings = pd.DataFrame({'article_id': vectors['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
            embeddings = pd.DataFrame(embeddings, columns=['article_id', 'x', 'y']).iloc[:, 1:].values

            # break out of the main loop if the results are optimal, ie. the centroids don't change their positions
            # much(more than our tolerance)
            centroid_changes = np.sum(np.abs(high_centroid_new - high_centroid), axis=1)
            assert centroid_changes.shape == (K,)
            max_centroid_change = np.max(centroid_changes)
            high_centroid = high_centroid_new
            low_centroid = low_centroid_new

            if max_centroid_change < self.tolerance:
                break
        # finalize
        points = umap.UMAP(metric='cosine', spread=30.0, init=points).fit_transform(
            vectors.iloc[:, 1:], y=best_group['country'])

        embeddings = pd.DataFrame({'article_id': vectors['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
        embeddings.to_csv(output_embedding)
        embeddings = pd.DataFrame(embeddings, columns=['article_id', 'x', 'y']).iloc[:, 1:].values

        mean_distance_high = get_mean_centroid_distance(data, high_centroid, best_group['country'], dimenstion="high")
        mean_distance_low = get_mean_centroid_distance(embeddings, low_centroid, best_group['country'], dimenstion="low")
        mean_distance_label = get_mean_centroid_distance(filtered_label_matrix, country_label, best_group['country'], dimenstion="high")

        return best_group, mean_distance_high, mean_distance_low, mean_distance_label, iterations
