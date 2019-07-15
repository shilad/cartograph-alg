import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from pygsp import graphs, filters

class Center:
    def __init__(self, id, position, cluster):
        self.id = id
        self.position = position
        self.coast = False
        self.cluster = cluster

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
    def __init__(self, points, xy_embedding_df, cluster_group_df):
        self.points = points
        self.num_points = len(points)
        self.augmented_points = self.add_water_points(points)
        print("points augmented")
        self.cluster_list = self.create_cluster(len(self.augmented_points), cluster_group_df)
        print("cluster_list done")
        self.is_cluster_preserved = self.denoise_cluster(self.augmented_points, len(set(self.cluster_list)))
        print('is_cluster_preserved done')
        self.cutting_index = 0 # the new index divides the original points and waterpoints
        print('start voronoi')
        self.vor = self.build_voronoi(self.augmented_points, self.is_cluster_preserved, self.cluster_list)
        print('voronoi end')
        self.centers_dic = {}
        self.edge_dic = {}
        self.corners_dic = {}

        print('start building graph')
        self.build_graph()
        self.draw_graph()

    def add_water_points(self, points):
        """
        Augment the original graph by adding waterpoints as boundaries and inner ponds
        :param points:
        :return:
        """
        water_level = 5
        max_abs_value = np.max(np.abs(points))

        def f(n):
            return (np.random.beta(0.8, 0.8, n) - 0.5) * 2 * (max_abs_value + 5)

        water_x_cor, water_y_cor = f(int(self.num_points * water_level)), f(int(self.num_points * water_level))
        num_squares = 5
        square_dot_sep = 2.0 * max_abs_value / (self.num_points * water_level)

        # Create nested squares around outside to prevent land masses from touching borders.
        for i in range(num_squares):
            n = np.arange(-max_abs_value, max_abs_value, square_dot_sep).shape[0]
            # coordinates for points on top, right, bottom, left of square
            square_x = np.concatenate([
                np.arange(-max_abs_value, max_abs_value, square_dot_sep),
                np.repeat(max_abs_value, n),
                np.arange(-max_abs_value, max_abs_value, square_dot_sep),
                np.repeat(-max_abs_value, n)])
            square_y = np.concatenate([
                np.repeat(max_abs_value, n),
                np.arange(-max_abs_value, max_abs_value, square_dot_sep),
                np.repeat(max_abs_value, n),
                np.arange(-max_abs_value, max_abs_value, square_dot_sep)])
            water_x_cor = np.concatenate([water_x_cor, square_x])
            water_y_cor = np.concatenate([water_y_cor, square_y])

        water_points = np.array(list(zip(water_x_cor, water_y_cor)))
        print(len(water_points))
        return np.concatenate([points, water_points])  # points after lengths are all waterpoints, [[x1, y1], [x2, y2] formats

    def create_cluster(self, length, cluster_groups_df):
        """
        Since the order in xy_embeddings.csv and cluster_groups.csv is the same, assign cluster to points taking advantage of the order
        :param length:
        :return:
        """
        cluster_list = []
        water_point_id = len(cluster_groups_df['country'].unique())
        for index in range(length):
            if index < self.num_points:
                cluster_list.append(int(cluster_groups_df.loc[index].iloc[0]))
            else:
                cluster_list.append(water_point_id)
        return cluster_list

    def denoise_cluster(self, points, num_cluster, tau=10):
        """
        Determine if the cluster after denoising is the same as the original
        :param points:
        :return: [boolean], false means cluster_id varies, true means cluster_id is preserved
        """
        length = len(points)
        graph = graphs.NNGraph(points, k=num_cluster)
        print("end creating")
        graph.estimate_lmax()
        filter = filters.Heat(graph, tau=tau)

        signal = np.empty(num_cluster * length).reshape(num_cluster, length)
        vectors = np.zeros(length * num_cluster).reshape(length, num_cluster)

        for i, vec in enumerate(vectors):
            vec[self.cluster_list[i]] = 1
        vectors = vectors.T
        for cluster_num, vec in enumerate(vectors):
            signal[cluster_num] = filter.analysis(vec)

        dominant_cluster = np.argmax(signal, axis=0)
        is_cluster_preserved = []
        print('start denoising')
        for i in range(length):
            if dominant_cluster[i] == int(self.cluster_df.iloc[i].loc[0]):
                is_cluster_preserved.append(True)
            else:
                is_cluster_preserved.append(False)
        print('end denoising')
        return is_cluster_preserved

    def build_voronoi(self, augmented_points, is_cluster_preserved, cluster_list):
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
        center = None
        if p in centers_dic:
            center = centers_dic[p]
        if center is None:
            center = Center(p, vor_points[p], self.cluster_list[p])  # since every point has an index now
            # if is_original:
            #     center = Center(p, vor_points[p], self.cluster_list[p])
            # else:
            #     center = Center(p, vor_points[p], -1)
        return center

    def initiate_corner(self, v, vor_vertices, corners_dic):
        corner = None
        if v in corners_dic:
            corner = corners_dic[v]
        if corner is None:
            corner = Corner(v, vor_vertices[v])
        return corner
    # def initiate_corner(self, v1, v2, vor_vertices, corners_dic):
    #     corner_1, corner_2 = None, None
    #     # Check if corners are added to the graph
    #     if v1 in corners_dic:
    #         corner_1 = corners_dic[v1]
    #     if v2 in corners_dic:
    #         corner_2 = corners_dic[v2]
    #     if corner_1 is None:
    #         corner_1 = Corner(v1, vor_vertices[v1])
    #     if corner_2 is None:
    #         corner_2 = Corner(v2, vor_vertices[v2])
    #     return corner_1, corner_2

    def build_graph(self):
        eps = sys.float_info.epsilon
        print(eps)
        vor = self.build_voronoi(self.points)
        if vor.points.shape[1] != 2:
            logging.warning('Required 2D input')
            return

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            # v1_coord, v2_coord = vor.vertices[v1], vor.vertices[v2]
            #
            # is_original_1 = False if p1 >= self.cutting_index else True
            # is_original_2 = False if p2 >= self.cutting_index else True
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
            # since scipy voronoi give one edge once, matters do not matter
            edge_id = len(self.edge_dic)
            if center_1.cluster != center_2.cluster:
                edge = Edge(len(self.edge_dic), center_1, center_2, corner_1, corner_2, True)
            # if is_original_1 ^ is_original_2:
            #     edge = Edge(len(self.edge_dic), center_1, center_2, corner_1, corner_2, True)
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



    # def in_box(self, points):
    #     max_coord = np.amax(points, axis=0)
    #     min_coord = np.amin(points, axis=0)
    #     bounding_box = np.array(
    #         [min_coord[0], max_coord[0], min_coord[1], max_coord[1]])  # [x_min, x_max, y_min, y_max]
    #     # bounding_box = np.array(
    #     #     [-100, 100, -100, 100])
    #     self.bounding_box = bounding_box
    #
    #     return np.logical_and(np.logical_and(bounding_box[0] <= points[:, 0],
    #                                          points[:, 0] <= bounding_box[1]),
    #                           np.logical_and(bounding_box[2] <= points[:, 1],
    #                                          points[:, 1] <= bounding_box[3]))

    # def build_voronoi(self, points):
    #     water_level = 5
    #     max_coord = np.amax(points, axis=0)
    #     min_coord = np.amin(points, axis=0)
    #     length = len(points)
    #     bounding_box = np.array(
    #         [min_coord[0], max_coord[0], min_coord[1], max_coord[1]])
    #     self.bounding_box = bounding_box
    #
    #     max_abs_value = np.max(np.abs(points))
    #
    #     def f(n):
    #         return (np.random.beta(0.8, 0.8, n) - 0.5) * 2 * (max_abs_value + 5)
    #
    #     water_x_cor, water_y_cor = f(int(length * water_level)), f(int(length * water_level))
    #     num_squares = 5
    #     square_dot_sep = 2.0 * max_abs_value / (length * water_level)
    #
    #     # Create nested squares around outside to prevent land masses from touching borders.
    #     for i in range(num_squares):
    #         n = np.arange(-max_abs_value, max_abs_value, square_dot_sep).shape[0]
    #         # coordinates for points on top, right, bottom, left of square
    #         square_x = np.concatenate([
    #             np.arange(-max_abs_value, max_abs_value, square_dot_sep),
    #             np.repeat(max_abs_value, n),
    #             np.arange(-max_abs_value, max_abs_value, square_dot_sep),
    #             np.repeat(-max_abs_value, n)])
    #         square_y = np.concatenate([
    #             np.repeat(max_abs_value, n),
    #             np.arange(-max_abs_value, max_abs_value, square_dot_sep),
    #             np.repeat(max_abs_value, n),
    #             np.arange(-max_abs_value, max_abs_value, square_dot_sep)])
    #         water_x_cor = np.concatenate([water_x_cor, square_x])
    #         water_y_cor = np.concatenate([water_y_cor, square_y])
    #
    #     water_points = np.array(list(zip(water_x_cor, water_y_cor)))
    #     points = np.concatenate([points, water_points])           # points after lengths are all waterpoints
    #
    #     vor = Voronoi(points)
    #     self.vor = vor
    #     return vor








    def draw_graph(self):
        # for id, center in self.centers_dic.items():
        #     print(center.)
        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                plt.plot([a[0], b[0]], [a[1], b[1]], 'ro-', marker='o', markersize=0.01)
        plt.show()






df = pd.read_csv("../../experiments/food/0009/xy_embeddings.csv")
df.x = df.x.round(6)
df.y = df.y.round(6)
points = np.zeros(shape=(df.shape[0], 2))
for index, row in df.iterrows():
    points[index] = [row['x'], row['y']]
print(points[1426][0])

clusters = pd.read_csv("../../experiments/food/0009/cluster_groups.csv")

g = Graph(points, df, clusters)

