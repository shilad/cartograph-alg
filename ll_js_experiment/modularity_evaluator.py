"""
Given 2D xy_embedding_csv file, evaluate the modularity
(dense connection between nodes within modules but sparse connection between modules)
High modularity means good community structure
"""

from sklearn.neighbors import NearestNeighbors
import pandas as pd
from igraph import *
import logging


def find_k_near_neighbors(x_y_embeddings_csv, k=10):
    """
    Given a 2-D csv file, find the k-nearest neighbors with 2 output csv file
    indices_dic: article_id : neighbors_with_id
    distance_dic: article_id : neighbors_distance

    :param x_y_embeddings_csv:
    :param k:
    :return:
    """

    df = pd.read_csv(x_y_embeddings_csv)

    feature_space = []

    indices_to_id = {}
    distance_dic = {}
    indices_dic = {}

    for index, row in df.iterrows():
        feature_space.append([row['x'], row['y']])
        indices_to_id.update({index:int(row['article_id'])})

    # print(feature_space)

    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)
    distance, indices = nbrs.kneighbors(feature_space)

    for row in range(0, indices.shape[0]):
        for col in range(0, indices.shape[1]):
            index = indices[row][col]
            indices[row][col] = indices_to_id[index]
        indices_dic.update({indices_to_id[row]: indices[row]})
        distance_dic.update({indices_to_id[row]: distance[row]})

    return distance_dic, indices_dic


def build_network(distance_dic, indices_dic):
    edges = []
    weights = []

    for key, value in indices_dic.items():
        for neighbor in range(1, len(value)):
            edges.append(tuple((key, value[neighbor])))
            weights.append(distance_dic[key][neighbor])

    G = Graph(edges)
    G.es['weight'] = weights
    return G


def calc_modularity(Graph):
    """
    Here We could choose 9 different measurement to test the modularity score
    The higher the better the cluster we produce
    Reference: https://yoyoinwanderland.github.io/2017/08/08/Community-Detection-in-Python/
    """
    vertex_clustering = Graph.community_multilevel(weights='weight')
    return Graph.modularity(vertex_clustering)


def main(map_directory, xy_embedding_csv):
    distance_dic, indices_dic = find_k_near_neighbors(map_directory + xy_embedding_csv)
    G = build_network(distance_dic, indices_dic)
    logging.warning("Modularity Score: %.6f", calc_modularity(G))


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    xy_embeddings_csv = sys.argv[2]
    main(map_directory, xy_embeddings_csv)

