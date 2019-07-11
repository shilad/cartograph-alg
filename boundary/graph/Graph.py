import logging
import numpy as np
import pandas as pd
from scipy.spatial import Voronoi


class Center:
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.coast = False

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
    def __init__(self, id, center1, center2, vertex1, vertex2):
        self.id = id
        self.d0 = center1
        self.d1 = center2
        self.v0 = vertex1
        self.v1 = vertex2


class Graph:
    def __init__(self, points):
        self.points = points
        self.centers_dic = {}
        self.edge_dic = {}
        self.corners_dic = {}
        self.build_graph()

    def initiate_center(self, p1, p2, vor_points, centers_dic):
        center_1, center_2 = None, None
        # Check if centers are added to the graph
        if p1 in centers_dic:
            center_1 = centers_dic[p1]
        if p2 in centers_dic:
            center_2 = centers_dic[p2]
        if center_1 is None:
            center_1 = Center(p1, vor_points[p1])
        if center_2 is None:
            center_2 = Center(p2, vor_points[p2])
        return center_1, center_2

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


    def build_graph(self):
        vor = Voronoi(self.points)
        if vor.points.shape[1] != 2:
            logging.warning('Required 2D input')
            return

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            center_1, center_2 = self.initiate_center(p1, p2, vor.points, self.centers_dic)
            # add neighboring voronoi polygon
            center_1.add_neighbor(center_2)
            center_2.add_neighbor(center_1)

            # add edges 2 points and 2 vertices
            # since scipy voronoi give one edge once, matters do not matter
            edge_id = len(self.edge_dic)
            edge = Edge(len(self.edge_dic), p1, p2, v1, v2)
            center_1.add_border(edge_id)
            center_2.add_border(edge_id)  # add edges' id

            corner_1, corner_2 = self.initiate_corner(v1, v2, vor.vertices, self.corners_dic)

            # add touches of a corner
            # the polygon centers p1, p2 touching the polygon corners v1, v2
            corner_1.add_touches(center_1.id)
            corner_1.add_touches(center_2.id)
            corner_2.add_touches(center_1.id)
            corner_2.add_touches(center_2.id)

            # add the edges extending from a corner
            corner_1.add_protrudes(edge_id)
            corner_2.add_protrudes(edge_id)

            # add adjacent corner of a corner
            corner_1.add_adjacent(corner_2.id)
            corner_2.add_adjacent(corner_1.id)

            # add adjacent corner of a corner
            center_1.add_corner(corner_1.id)
            center_1.add_corner(corner_2.id)
            center_2.add_corner(corner_1.id)
            center_2.add_corner(corner_2.id)

            self.centers_dic.update({p1: center_1})
            self.centers_dic.update({p2: center_2})
            self.corners_dic.update({v1: corner_1})
            self.corners_dic.update({v2: corner_2})
            self.edge_dic.update({edge_id: edge})


points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])

g = Graph(points)

for k, v in g.centers_dic.items():
    print(k)
    center = v
    for n in center.neighbors:
        print(n.id)
    print("-------------")

#
# points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
# vor = Voronoi(points)
#
#
# def read_points(xy_embedding_csv):
#     df = pd.read_csv(xy_embedding_csv)
#     points = np.zeros(shape=(df.shape[0], 2))
#     for index, row in df.iterrows():
#         points[index] = [row['x'], row['y']]
#
#     return points
#
#
# def initiate_center(p1, p2, vor_points, centers_dic):
#     center_1, center_2 = None, None
#     # Check if centers are added to the graph
#     if p1 in centers_dic:
#         center_1 = centers_dic[p1]
#     if p2 in centers_dic:
#         center_2 = centers_dic[p2]
#     if center_1 is None:
#         center_1 = Center(p1, vor_points[p1])
#     if center_2 is None:
#         center_2 = Center(p2, vor_points[p2])
#     return center_1, center_2
#
#
# def initiate_corner(v1, v2, vor_vertices, corners_dic):
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
#
#
# def build_graph(xy_embedding_csv):
#     #points = read_points(xy_embedding_csv)
#     points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2]])
#
#     vor = Voronoi(points)
#
#     if vor.points.shape[1] != 2:
#         logging.warning('Required 2D input')
#         return
#
#     centers_dic = {}
#     edge_dic = {}
#     corners_dic = {}
#
#     for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
#         center_1, center_2 = initiate_center(p1, p2, vor.points, centers_dic)
#         # add neighboring voronoi polygon
#         center_1.add_neighbor(center_2)
#         center_2.add_neighbor(center_1)
#
#         # add edges 2 points and 2 vertices
#         # since scipy voronoi give one edge once, matters do not matter
#         edge_id = len(edge_dic)
#         edge = Edge(len(edge_dic), p1, p2, v1, v2)
#         center_1.add_border(edge_id)
#         center_2.add_border(edge_id)  # add edges' id
#
#         corner_1, corner_2 = initiate_corner(v1, v2, vor.vertices, corners_dic)
#
#         # add touches of a corner
#         # the polygon centers p1, p2 touching the polygon corners v1, v2
#         corner_1.add_touches(center_1.id)
#         corner_1.add_touches(center_2.id)
#         corner_2.add_touches(center_1.id)
#         corner_2.add_touches(center_2.id)
#
#         # add the edges extending from a corner
#         corner_1.add_protrudes(edge_id)
#         corner_2.add_protrudes(edge_id)
#
#         # add adjacent corner of a corner
#         corner_1.add_adjacent(corner_2.id)
#         corner_2.add_adjacent(corner_1.id)
#
#         # add adjacent corner of a corner
#         center_1.add_corner(corner_1.id)
#         center_1.add_corner(corner_2.id)
#         center_2.add_corner(corner_1.id)
#         center_2.add_corner(corner_2.id)
#
#
#         centers_dic.update({p1: center_1})
#         centers_dic.update({p2: center_2})
#         corners_dic.update({v1: corner_1})
#         corners_dic.update({v2: corner_2})
#         edge_dic.update({edge_id: edge})
#
#     # for k, v in edge_dic.items():
#     #     print(k)
#     #     edge = v
#     #     print(str(edge.d0) + " " + str(edge.d1) + " " + str(edge.v0) + " " + str(edge.v1))
#     #
#     # for k, v in corners_dic.items():
#     #     print(k)
#     #     corner = v
#     #     for n in corner.protrudes:
#     #         print(n)
#     #     print("-------------")
#
#     for k, v in corners_dic.items():
#         print(k)
#         corner = v
#         for n in corner.adjacent:
#             print(n)
#         print("-------------")
# build_graph(" ")
#
#
# centers = []
# corners = []
# edges = []