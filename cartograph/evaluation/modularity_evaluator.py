"""
Given 2D xy_embedding_csv file, evaluate the modularity,
We want dense connection between nodes within modules but sparse connection between modules
High modularity corresponds to good community structure

The bash input should be map_directory, xy_embeddings_csv, cluster_groups_csv,
and network construction method 'nn' or 'distance'

To install igraph
brew install pkg-config
brew install igraph
sudo pip install python-igraph

Author: Yuren 'Rock' Pang
"""

from sklearn.neighbors import NearestNeighbors
import pandas as pd
from igraph import *
import logging
import json


def preprocess(x_y_embeddings_csv):
    df = pd.read_csv(x_y_embeddings_csv)
    feature_space = []
    indices_to_id = {}

    for index, row in df.iterrows():
        feature_space.append([row['x'], row['y']])
        indices_to_id.update({index:int(row['article_id'])})

    return feature_space, indices_to_id


def find_k_near_neighbors(feature_space, k=15):
    """
    Given a 2-D csv file, find the k-nearest neighbors with 2 output csv file
    indices_dic: article_id : neighbors_with_id
    distance_dic: article_id : neighbors_distance

    :param x_y_embeddings_csv:
    :param k:
    :return:
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(feature_space)

    return nbrs.kneighbors(feature_space)  # the output is two lists: distance and indices


def find_neighbors_within_d_distance(feature_space, d = 3):
    distances_lst = []
    neighbors_lst = []

    for center in range(len(feature_space)):
        center_x = feature_space[center][0]
        center_y = feature_space[center][1]

        distances = [0]
        neighbors = [center]

        for point in range(len(feature_space)):
            if center == point:
                continue
            x = feature_space[point][0]
            y = feature_space[point][1]
            a2b2 = (x-center_x) * (x-center_x) + (y-center_y) * (y-center_y)
            if a2b2 <= d * d:
                neighbors.append(point)
                distances.append(math.sqrt(a2b2))
        neighbors_lst.append(neighbors)
        distances_lst.append(distances)

    return distances_lst, neighbors_lst


def build_network(distances_lst, neighbors_lst, indices_to_id):
    edges = []
    weights = []

    for node in range(0, len(neighbors_lst)):  # default id, not the article id !!
        node_list = neighbors_lst[node]
        for neighbor in range(1, len(node_list)):
            if len(node_list) <= 1:
                continue
            else:
                edges.append(tuple((node_list[0], node_list[neighbor])))
                weights.append(distances_lst[node][neighbor])

    G = Graph()
    G.add_vertices(len(neighbors_lst))

    names = [indices_to_id[index] for index in G.vs.indices]
    G.vs['name'] = names

    G.add_edges(edges)
    G.es['weight'] = weights

    return G.simplify(combine_edges=max)


def calc_modularity(Graph, cluster_groups_csv):
    """
    Here We could choose 9 different measurement to test the modularity score
    The higher the better the cluster we produce
    Reference: https://yoyoinwanderland.github.io/2017/08/08/Community-Detection-in-Python/
    """
    cluster_groups = pd.read_csv(cluster_groups_csv)
    country = cluster_groups['country'].tolist()

    #vertex_clustering = Graph.community_multilevel(weights='weight')
    return Graph.modularity(country)


# feature_space, indices_to_id = preprocess("../data/tech/xy_embeddings.csv")
# distance_lst, indices_lst = find_k_near_neighbors(feature_space)
# G = build_network(distance_lst, indices_lst, indices_to_id)
# mod_score = calc_modularity(G, "../data/tech/cluster_groups.csv")
# print(str(json.dumps({'modularity':mod_score})))
# print("Modularity Score: %.6f", calc_modularity(G, "../data/tech/cluster_groups.csv"))


def main(map_directory, xy_embedding_csv, cluster_groups_csv, method='nn'):
    feature_space, indices_to_id = preprocess(map_directory + "/" + xy_embedding_csv)

    if method == 'nn':
        distance_lst, indices_lst = find_k_near_neighbors(feature_space)
    else:
        distance_lst, indices_lst = find_neighbors_within_d_distance(feature_space)

    G = build_network(distance_lst, indices_lst, indices_to_id)
    mod_score = calc_modularity(G, map_directory + cluster_groups_csv)
    print(str(json.dumps({'modularity':mod_score})))
    # logging.warning("Modularity Score: %.6f", mod_score)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s map_directory xy_embeddings.csv cluster_group_csv method (nn)' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    xy_embeddings_csv = sys.argv[2]
    cluster_groups_csv = sys.argv[3]
    method = sys.argv[4]
    main(map_directory, xy_embeddings_csv, cluster_groups_csv, method=method)

