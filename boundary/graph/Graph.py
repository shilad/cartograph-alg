import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi


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
    def __init__(self, points, xy_embedding_df, cluster_df):
        self.points = points
        self.xy_embedding_df = xy_embedding_df
        self.cluster_df = cluster_df

        self.centers_dic = {}
        self.edge_dic = {}
        self.corners_dic = {}
        self.vor = None
        self.bounding_box = []

        self.build_graph()

        self.draw_graph()

    def initiate_center(self, p, vor_points, centers_dic, is_original):
        center = None
        if p in centers_dic:
            center = centers_dic[p]
        if center is None:
            if is_original:
                center = Center(p, vor_points[p], self.find_cluster(points[p][0], points[p][1]))
            else:
                center = Center(p, vor_points[p], -1)
        return center

    def initiate_corner(self, v1, v2, vor_vertices, corners_dic):
        corner_1, corner_2 = None, None
        # Check if corners are added to the graph
        if v1 in corners_dic:
            corner_1 = corners_dic[v1]
        if v2 in corners_dic:
            corner_2 = corners_dic[v2]
        if corner_1 is None:
            corner_1 = Corner(v1, vor_vertices[v1])
        if corner_2 is None:
            corner_2 = Corner(v2, vor_vertices[v2])
        return corner_1, corner_2

    def find_cluster(self, x, y):
        article_id = self.xy_embedding_df.loc[(self.xy_embedding_df['x'] == x) & (self.xy_embedding_df['y'] == y), ['article_id']].iloc[0,:].to_list()[0]
        return self.cluster_df.loc[self.cluster_df['article_id'] == article_id, ['country']].iloc[0, :].to_list()[0]

    def in_box(self, points):
        max_coord = np.amax(points, axis=0)
        min_coord = np.amin(points, axis=0)
        # bounding_box = np.array(
        #     [min_coord[0], max_coord[0], min_coord[1], max_coord[1]])  # [x_min, x_max, y_min, y_max]
        bounding_box = np.array(
            [-100, 100, -100, 100])
        self.bounding_box = bounding_box

        return np.logical_and(np.logical_and(bounding_box[0] <= points[:, 0],
                                             points[:, 0] <= bounding_box[1]),
                              np.logical_and(bounding_box[2] <= points[:, 1],
                                             points[:, 1] <= bounding_box[3]))

    def build_voronoi(self, points):
        inside = self.in_box(points)

        # Mirror points
        points_center = points
        points_left = np.copy(points_center)
        points_left[:, 0] = self.bounding_box[0] - (points_left[:, 0] - self.bounding_box[0])
        points_right = np.copy(points_center)
        points_right[:, 0] = self.bounding_box[1] + (self.bounding_box[1] - points_right[:, 0])
        points_down = np.copy(points_center)
        points_down[:, 1] = self.bounding_box[2] - (points_down[:, 1] - self.bounding_box[2])
        points_up = np.copy(points_center)
        points_up[:, 1] = self.bounding_box[3] + (self.bounding_box[3] - points_up[:, 1])

        points = np.append(points_center, np.append(np.append(points_left, points_right, axis=0),
                                        np.append(points_down, points_up, axis=0),
                                        axis=0), axis=0)
        vor = Voronoi(points)
        self.vor = vor
        return vor

    def build_graph(self):
        eps = sys.float_info.epsilon
        print(eps)
        vor = self.build_voronoi(self.points)
        if vor.points.shape[1] != 2:
            logging.warning('Required 2D input')
            return

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            v1_coord, v2_coord = vor.vertices[v1], vor.vertices[v2]

            if self.bounding_box[0] - eps > v1_coord[0] or v1_coord[0] > self.bounding_box[1] + eps or \
                    self.bounding_box[2] - eps > v1_coord[1] or v1_coord[1] > self.bounding_box[3] + eps or \
                    self.bounding_box[0] - eps > v2_coord[0] or v2_coord[0] > self.bounding_box[1] + eps or \
                    self.bounding_box[2] - eps > v2_coord[1] or v2_coord[1] > self.bounding_box[3] + eps:
                continue

            is_original_1 = False if p1 > 4099 else True
            is_original_2 = False if p2 > 4099 else True
            center_1 = self.initiate_center(p1, vor.points, self.centers_dic, is_original_1)
            center_2 = self.initiate_center(p2, vor.points, self.centers_dic, is_original_2)

            # add neighboring voronoi polygon
            center_1.add_neighbor(center_2)
            center_2.add_neighbor(center_1)

            corner_1, corner_2 = self.initiate_corner(v1, v2, vor.vertices, self.corners_dic)

            # add touches of a corner
            # the polygon centers p1, p2 touching the polygon corners v1, v2
            corner_1.add_touches(center_1)
            corner_1.add_touches(center_2)
            corner_2.add_touches(center_1)
            corner_2.add_touches(center_2)

            # add edges 2 points and 2 vertices
            # since scipy voronoi give one edge once, matters do not matter
            edge_id = len(self.edge_dic)
            if is_original_1 ^ is_original_2:
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

    def draw_graph(self):
        # for id, center in self.centers_dic.items():
        #     print(center.)
        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                plt.plot([a[0], b[0]], [a[1], b[1]], 'ro-', marker='o')
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

