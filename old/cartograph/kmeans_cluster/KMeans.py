from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import numpy as np
import warnings
import random
from heapq import *

class KMeans:
    def __init__(self, k=7, tolerance=0.00001, max_iterations=500):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.centroids = {}

    def cal_label_dim_distance(self):
        return ""

    def fit_joint_all(self, article_vectors, article_ids, low_dim_embeddings, article_labels, tf_idf_score=None, weight=0.05):
        """
        :param article_vectors: pd df 100-dimensional vectors
        :param article_ids:
        :param low_dim_embeddings:
        :param tf_idf_score:
        :param weight:
        :return:
        """
        high_centroids = np.stack(article_vectors[:self.k])
        low_centroids = np.stack(low_dim_embeddings[:self.k])

        for i in range(self.max_iterations):
            high_dim_distance = cosine_distances(article_vectors, high_centroids)
            low_dim_distance = euclidean_distances(low_dim_embeddings, low_centroids)
            label_dim_distance = self.cal_label_dim_distance()

            xy_range = (np.max(low_dim_embeddings) - np.min(low_dim_embeddings))
            max_distance = np.sqrt(xy_range * xy_range + xy_range * xy_range)
            low_dim_distance /= max_distance


            cluster_label_matrix = [[]]   # 7 * 13000 cluster matrix
            article_label_matrix = [[]]   # 4051 * 13000 article matrix

            label_score = 0
            pq = []
            while len(pq) < 20:
                heappush(pq, (id, label_score))

            




