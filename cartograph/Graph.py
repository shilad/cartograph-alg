"""
Given the xy_embeddings.csv and cluster_groups.csv, draw the final polygon maps with borders

Author: Yuren 'Rock' Pang
Reference:  For denoising function: https://github.com/shilad/cartograph/blob/develop/cartograph/Denoiser.py
"""
import logging
import drawSvg as draw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from pygsp import graphs, filters


class Center:
    def __init__(self, id, position, cluster, article_id):
        self.id = id
        self.position = position
        self.coast = False
        self.cluster = cluster
        self.article_id = article_id

        self.neighbors = set()  # neighboring Voronoi centers
        self.border = set()    # voronoi bordering edges
        self.corners = set()   # voronoi polygon corners

    def add_neighbor(self, center):
        self.neighbors.add(center)

    def add_border(self, edge):
        self.border.add(edge)

    def add_corner(self, corner):
        self.corners.add(corner)


class Corner:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.coast = False

        self.touches = set()    # set of Center(polygon) touching this corner
        self.protrudes = set()  # a set of edges touching the corner
        self.adjacent = set()   # a set of corners connected to this one

    def add_touches(self, center):
        self.touches.add(center)

    def add_protrudes(self, edge):
        self.protrudes.add(edge)

    def add_adjacent(self, corner):
        self.adjacent.add(corner)


class Edge:
    def __init__(self, id, center1, center2, vertex1, vertex2, is_border):
        self.id = id
        self.d0 = center1
        self.d1 = center2
        self.v0 = vertex1
        self.v1 = vertex2
        self.is_border = is_border


class Graph:
    def __init__(self, xy_embedding_csv, cluster_group_csv):
        self.points, self.cluster_group_df = self.preprocess_file(xy_embedding_csv, cluster_group_csv)
        if self.points.shape[1] != 2:
            logging.warning('Required 2D input')
            return
        self.num_points = len(self.points)
        self.augmented_points = self.add_water_points(self.points)
        self.cluster_list = []
        self.article_id_list = []
        self.create_cluster_and_article_id(len(self.augmented_points), self.cluster_group_df)

        self.is_cluster_preserved = self.denoise_cluster(self.augmented_points, len(set(self.cluster_list)))
        self.cutting_index = 0  # the new index divides the original points and waterpoints
        self.vor = self.build_voronoi(self.augmented_points, self.is_cluster_preserved, self.cluster_list)

        self.centers_dic = {}
        self.edge_dic = {}
        self.corners_dic = {}

        self.build_graph()

    def preprocess_file(self, xy_embedding_csv, cluster_group_csv):
        xy_embedding_df = pd.read_csv(xy_embedding_csv)
        cluster_groups_def = pd.read_csv(cluster_group_csv)
        xy_embedding_df.x.round(6)
        xy_embedding_df.y.round(6)
        points = np.zeros(shape=(xy_embedding_df.shape[0], 2))

        for index, row in xy_embedding_df.iterrows():
            points[index] = [row['x'], row['y']]
        return points, cluster_groups_def

    def add_water_points(self, points):
        """
        Augment the original graph by adding waterpoints as boundaries and inner ponds
        :param points:
        :return: [[x1, y1], [x2, y2], ...] all points on the map including waterpoints and original points
        """
        # Drop waterpoints inside the original dots
        water_level = 0.0005
        max_abs_value = np.max(np.abs(points))
        def f(n):
            return (np.random.beta(0.8, 0.8, n) - 0.5) * 2 * (max_abs_value + 5)
        water_x_cor, water_y_cor = f(int(self.num_points * water_level)), f(int(self.num_points * water_level))

        # Drop waterpoint boundaries and constrain the map
        max_coord = np.amax(points, axis=0)
        min_coord = np.amin(points, axis=0)
        BUFFER = 30
        bounding_box = np.array(
            [min_coord[0]-BUFFER, max_coord[0]+BUFFER, min_coord[1]-BUFFER, max_coord[1]+BUFFER])  # [x_min, x_max, y_min, y_max]

        num_squares, square_dot_sep = 1, 2  # the number of boundaries and the distance between boundary dots

        # Construct the boundaries dots
        for i in range(num_squares):
            num_of_points_x = np.arange(bounding_box[0], bounding_box[1], square_dot_sep).shape[0]
            # coordinates for points on top, right, bottom, left of square
            square_x = np.concatenate([
                np.arange(bounding_box[0], bounding_box[1], square_dot_sep),
                np.repeat(bounding_box[1], num_of_points_x),
                np.arange(bounding_box[0], bounding_box[1], square_dot_sep),
                np.repeat(bounding_box[0], num_of_points_x)])
            num_of_points_y = np.arange(bounding_box[2], bounding_box[3], square_dot_sep).shape[0]
            square_y = np.concatenate([
                np.repeat(bounding_box[2], num_of_points_y),
                np.arange(bounding_box[2], bounding_box[3], square_dot_sep),
                np.repeat(bounding_box[3], num_of_points_y),
                np.arange(bounding_box[2], bounding_box[3], square_dot_sep)])
            water_x_cor = np.concatenate([water_x_cor, square_x])
            water_y_cor = np.concatenate([water_y_cor, square_y])
        water_points = np.array(list(zip(water_x_cor, water_y_cor)))

        return np.concatenate([points, water_points])

    def create_cluster_and_article_id(self, length, cluster_groups_df):
        """
        Since the order in xy_embeddings.csv and cluster_groups.csv is the same, assign cluster to points taking advantage of the order
        Mutate the cluster and article_id look-up lists
        """
        water_point_id = len(cluster_groups_df['country'].unique())
        for index in range(length):
            if index < self.num_points:
                self.cluster_list.append(int(cluster_groups_df.loc[index].iloc[0]))
                self.article_id_list.append(int(cluster_groups_df.loc[index].iloc[1]))
            else:
                self.cluster_list.append(water_point_id)
                self.article_id_list.append(-1)

    def denoise_cluster(self, points, num_cluster, tau=10):
        """
        Determine if the cluster after denoising is the same as the original
        :param points:
        :return: [boolean], false means cluster_id varies, true means cluster_id is preserved
        """
        length = len(points)
        graph = graphs.NNGraph(points, k=num_cluster)
        graph.estimate_lmax()
        fn = filters.Heat(graph, tau=tau)

        signal = np.empty(num_cluster * length).reshape(num_cluster, length)   # create num_cluster*len(points) matrix
        vectors = np.zeros(length * num_cluster).reshape(length, num_cluster)  # create len(points)*num_cluster matrix

        # fill the vectors sparse matrix
        for i, vec in enumerate(vectors):
            vec[self.cluster_list[i]] = 1
        vectors = vectors.T

        # fill the denoising matrix, find the dominant cluster of each points
        for cluster_num, vec in enumerate(vectors):
            signal[cluster_num] = fn.analyze(vec)

        # see if the dominant cluster after denoising is the same as the original cluster
        dominant_cluster = np.argmax(signal, axis=0)
        is_cluster_preserved = []
        for i in range(length):
            if dominant_cluster[i] == int(self.cluster_list[i]):
                is_cluster_preserved.append(True)
            else:
                is_cluster_preserved.append(False)
        return is_cluster_preserved

    def build_voronoi(self, augmented_points, is_cluster_preserved, cluster_list):
        """
        Draw Voronoi diagram of the points whose clusters didn't change after denoising function
        Update the cutting points before which are the points in the original graph and their cluster didn't change
        :param augmented_points:
        :param is_cluster_preserved:
        :param cluster_list:
        :return:
        """
        vor_points, vor_clusters = [], []
        cutting_index = 0
        for index, coord in enumerate(augmented_points):
            if is_cluster_preserved[index]:
                vor_points.append(coord)
                vor_clusters.append(cluster_list[index])
            if index < self.num_points:
                cutting_index = cutting_index + 1
        self.cluster_list = vor_clusters  # since order might have changed, the indexing in cluster_list must be updated
        self.cutting_index = cutting_index
        return Voronoi(vor_points)

    def initiate_center(self, p, vor_points, centers_dic):
        if p in centers_dic:
            return centers_dic[p]
        return Center(p, vor_points[p], self.cluster_list[p], self.article_id_list[p])

    def initiate_corner(self, v, vor_vertices, corners_dic):
        if v in corners_dic:
            return corners_dic[v]
        return Corner(v, vor_vertices[v])

    def build_graph(self):
        vor = self.vor

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            center_1 = self.initiate_center(p1, vor.points, self.centers_dic)
            center_2 = self.initiate_center(p2, vor.points, self.centers_dic)

            # add neighboring voronoi polygon
            center_1.add_neighbor(center_2)
            center_2.add_neighbor(center_1)

            corner_1 = self.initiate_corner(v1, vor.vertices, self.corners_dic)
            corner_2 = self.initiate_corner(v2, vor.vertices, self.corners_dic)
            # corner_1, corner_2 = self.initiate_corner(v1, v2, vor.vertices, self.corners_dic)

            # add touches of a corner
            # the polygon centers p1, p2 touching the polygon corners v1, v2
            corner_1.add_touches(center_1)
            corner_1.add_touches(center_2)
            corner_2.add_touches(center_1)
            corner_2.add_touches(center_2)

            # add edges 2 points and 2 vertices
            edge_id = len(self.edge_dic)
            if center_1.cluster != center_2.cluster:
                edge = Edge(len(self.edge_dic), center_1, center_2, corner_1, corner_2, True)
            else:
                edge = Edge(len(self.edge_dic), center_1, center_2, corner_1, corner_2, False)

            center_1.add_border(edge)
            center_2.add_border(edge)  # add edges' id

            # add the edges extending from a corner
            corner_1.add_protrudes(edge)
            corner_2.add_protrudes(edge)

            # add adjacent corner of a corner
            corner_1.add_adjacent(corner_2)
            corner_2.add_adjacent(corner_1)

            # add adjacent corner of a corner
            center_1.add_corner(corner_1)
            center_1.add_corner(corner_2)
            center_2.add_corner(corner_1)
            center_2.add_corner(corner_2)

            self.centers_dic.update({p1: center_1})
            self.centers_dic.update({p2: center_2})
            self.corners_dic.update({v1: corner_1})
            self.corners_dic.update({v2: corner_2})
            self.edge_dic.update({edge_id: edge})

    def export_clusters(self, directory):
        row_list = []
        for id, center in self.centers_dic.items():
            if center.article_id != -1:
                row_list.append({'country': center.cluster, 'article_id': center.article_id})
        pd.DataFrame(row_list).to_csv(directory + "/cluster_groups.csv", index=False)

    def export_boundaries(self, directory):
        row_list = []
        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                row_list.append({'x1': a[0], 'y1': a[1], 'x2': b[0], 'y2': b[1]})
        pd.DataFrame(row_list).to_csv(directory + "/boundary.csv", index=False)

    def draw_graph(self):
        colors = ['pink', 'yellow','green', 'red', 'orange', 'grey', 'purple', 'brown', 'blue']
        for id, center in self.centers_dic.items():
            color = colors[center.cluster]
            plt.plot(center.position[0], center.position[1], 'o', color=color)

        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                plt.plot([a[0], b[0]], [a[1], b[1]], 'ro-', marker='o', markersize=0.01)

        plt.show()


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    g = Graph(map_directory + '/xy_embeddings.csv', map_directory + '/cluster_groups.csv')
    g.export_boundaries(map_directory)
