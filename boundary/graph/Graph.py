import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
from pygsp import graphs, filters
from boundary.graph import Center, Corner, Edge


class Graph:
    def __init__(self, points, cluster_group_df):
        self.points = points
        self.num_points = len(points)
        self.augmented_points = self.add_water_points(points)
        self.cluster_list = []
        self.article_id_list = []
        self.create_cluster_and_article_id(len(self.augmented_points), cluster_group_df)

        self.is_cluster_preserved = self.denoise_cluster(self.augmented_points, len(set(self.cluster_list)))
        self.cutting_index = 0 # the new index divides the original points and waterpoints
        self.vor = self.build_voronoi(self.augmented_points, self.is_cluster_preserved, self.cluster_list)

        self.centers_dic = {}
        self.edge_dic = {}
        self.corners_dic = {}

        self.build_graph()

    def add_water_points(self, points):
        """
        Augment the original graph by adding waterpoints as boundaries and inner ponds
        :param points:
        :return:
        """
        water_level = 0.0005
        max_abs_value = np.max(np.abs(points))
        max_coord = np.amax(points, axis=0)
        min_coord = np.amin(points, axis=0)
        bounding_box = np.array(
            [-100, 100, -100, 100])  # [x_min, x_max, y_min, y_max]

        def f(n):
            return (np.random.beta(0.8, 0.8, n) - 0.5) * 2 * (max_abs_value + 5)

        water_x_cor, water_y_cor = f(int(self.num_points * water_level)), f(int(self.num_points * water_level))

        num_squares = 1
        square_dot_sep = 2
        # square_dot_sep = max_abs_value / (self.num_points * water_level)

        # Create nested squares around outside to prevent land masses from touching borders.
        for i in range(num_squares):
            n = np.arange(bounding_box[0], bounding_box[1], square_dot_sep).shape[0]
            # coordinates for points on top, right, bottom, left of square
            square_x = np.concatenate([
                np.arange(bounding_box[0], bounding_box[1], square_dot_sep),
                np.repeat(bounding_box[1], n),
                np.arange(bounding_box[0], bounding_box[1], square_dot_sep),
                np.repeat(bounding_box[0], n)])
            square_y = np.concatenate([
                np.repeat(bounding_box[2], n),
                np.arange(bounding_box[2], bounding_box[3], square_dot_sep),
                np.repeat(bounding_box[3], n),
                np.arange(bounding_box[2], bounding_box[3], square_dot_sep)])
            water_x_cor = np.concatenate([water_x_cor, square_x])
            water_y_cor = np.concatenate([water_y_cor, square_y])

        water_points = np.array(list(zip(water_x_cor, water_y_cor)))
        print(len(water_points))
        return np.concatenate([points, water_points])  # points after lengths are all waterpoints, [[x1, y1], [x2, y2] formats

    def create_cluster_and_article_id(self, length, cluster_groups_df):
        """
        Since the order in xy_embeddings.csv and cluster_groups.csv is the same, assign cluster to points taking advantage of the order
        :param length:
        :return:
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

        signal = np.empty(num_cluster * length).reshape(num_cluster, length)
        vectors = np.zeros(length * num_cluster).reshape(length, num_cluster)

        for i, vec in enumerate(vectors):
            vec[self.cluster_list[i]] = 1
        vectors = vectors.T
        for cluster_num, vec in enumerate(vectors):
            signal[cluster_num] = fn.analyze(vec)

        dominant_cluster = np.argmax(signal, axis=0)
        is_cluster_preserved = []
        print('start denoising')
        for i in range(length):
            if dominant_cluster[i] == int(self.cluster_list[i]):
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
            center = Center(p, vor_points[p], self.cluster_list[p], self.article_id_list[p])  # since every point has an index now
        return center

    def initiate_corner(self, v, vor_vertices, corners_dic):
        corner = None
        if v in corners_dic:
            corner = corners_dic[v]
        if corner is None:
            corner = Corner(v, vor_vertices[v])
        return corner

    def build_graph(self):
        vor = self.vor
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

    def export_clusters(self, path=''):
        row_list = []
        for id, center in self.centers_dic.items():
            if center.article_id != -1:
                row_list.append({'country':center.cluster, 'article_id':center.article_id})
        pd.DataFrame(row_list).to_csv(path + "cluster_groups.csv", index=False)

    def export_boundaries(self, path):
        row_list = []
        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                row_list.append({'x1': a[0], 'y1': a[1], 'x2': b[0], 'y2': b[1]})
        pd.DataFrame(row_list).to_csv(path + "boundary.csv", index=False)


    def draw_graph(self):
        # for id, center in self.centers_dic.items():
        #     print(center.)
        colors = ['pink', 'yellow','green', 'red', 'orange', 'grey', 'purple', 'brown', 'blue']
        for id, center in self.centers_dic.items():
            color = colors[center.cluster]
            plt.plot(center.position[0], center.position[1], 'o', color=color)

        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                plt.plot([a[0], b[0]], [a[1], b[1]], 'ro-', marker='o', markersize=0.01)

        plt.show()






# df = pd.read_csv("../../experiments/food/0009/xy_embeddings.csv")
# df.x = df.x.round(6)
# df.y = df.y.round(6)
# points = np.zeros(shape=(df.shape[0], 2))
# for index, row in df.iterrows():
#     points[index] = [row['x'], row['y']]
# print(points[1426][0])
#
# clusters = pd.read_csv("../../experiments/food/0009/cluster_groups.csv")
#
# g = Graph(points, clusters)
# g.export_boundaries('')
# g.export_clusters('')

