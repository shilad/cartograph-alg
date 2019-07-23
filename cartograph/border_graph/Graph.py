"""
Given the xy_embeddings.csv and cluster_groups.csv, draw the final polygon maps with borders

Author: Yuren 'Rock' Pang
Reference:  For denoising function: https://github.com/shilad/cartograph/blob/develop/cartograph/Denoiser.py
"""
import random
import queue

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.spatial import Voronoi
from pygsp import graphs, filters
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from cartograph.border_graph.Center import Center
from cartograph.border_graph.Corner import Corner
from cartograph.border_graph.Edge import Edge


class Graph:
    def __init__(self, points, cluster_list, article_id_list, color_palette='hls'):
        self.points = points
        self.cluster_list = cluster_list
        self.article_id_list = article_id_list
        self.color_palette = color_palette
        self.bounding_box = self.create_bounding_box(points)

        # add water points to add randomness and inner lakes
        # build initial Voronoi barring the outlier points
        self.num_points = len(self.points)
        self.add_water_points(self.points)
        self.denoise_cluster(self.points, len(set(self.cluster_list)))
        self.vor = Voronoi(self.points)

        # dictionary to store all centers, corners, edges in the graph
        self.centers_dic = {}
        self.edge_dic = {}
        self.corners_dic = {}

        self.build_graph()

        # calculate elevation for color gradient changes and 'naturalize' Voronoi borders
        self.max_elevation = 0  # track the max elevation for the color gradient
        self.assign_elevation()
        self.clockwise_all_corners()
        self.noise_border()

    def create_bounding_box(self, points):
        max_coord = np.amax(points, axis=0)
        min_coord = np.amin(points, axis=0)
        BUFFER = 30
        return np.array([min_coord[0] - BUFFER, max_coord[0] + BUFFER, min_coord[1] - BUFFER, max_coord[1] + BUFFER])

    def initiate_center(self, p, vor_points, centers_dic):
        if p in centers_dic:
            return centers_dic[p]
        is_water = True if self.cluster_list[p] == len(set(self.cluster_list))-1 else False
        return Center(p, vor_points[p], self.cluster_list[p], self.article_id_list[p], is_water)

    def initiate_corner(self, v, vor_vertices, corners_dic):
        if v in corners_dic:
            return corners_dic[v]
        return Corner(v, vor_vertices[v])

    def add_water_points(self, points):
        """
        Augment the original graph by adding waterpoints as boundaries and inner ponds
        :param points:
        :return: [[x1, y1], [x2, y2], ...] all points on the map including waterpoints and original points
        """
        water_level = 0.2
        water_point_dis = 1
        num_boundary = 5
        acc = 0     # extend the outside boundary again and again
        max_abs_value = np.max(np.abs(points))
        num_points = len(points)
        bounding_box = self.bounding_box

        def f(n):
            return (np.random.beta(0.8, 0.8, n) - 0.5) * 2 * (max_abs_value + 5)

        water_x_cor, water_y_cor = f(int(self.num_points * water_level)), f(int(self.num_points * water_level))

        # coordinates for points on top, right, bottom, left of square
        # num_of_points are the points are the num of waterpoints on sqaure boundaries
        for i in range(num_boundary):
            num_of_points_x = np.arange(bounding_box[0]-acc, bounding_box[1]+acc, water_point_dis).shape[0]
            num_of_points_y = np.arange(bounding_box[2]-acc, bounding_box[3]+acc, water_point_dis).shape[0]
            square_x = np.concatenate([
                np.arange(bounding_box[0]-acc, bounding_box[1]+acc, water_point_dis),
                np.repeat(bounding_box[1]+acc, num_of_points_y),
                np.arange(bounding_box[0]-acc, bounding_box[1]+acc, water_point_dis),
                np.repeat(bounding_box[0]-acc, num_of_points_y)])
            square_y = np.concatenate([
                np.repeat(bounding_box[2]-acc, num_of_points_x),
                np.arange(bounding_box[2]-acc, bounding_box[3]+acc, water_point_dis),
                np.repeat(bounding_box[3]+acc, num_of_points_x),
                np.arange(bounding_box[2]-acc, bounding_box[3]+acc, water_point_dis)])

            water_x_cor = np.concatenate([water_x_cor, square_x])
            water_y_cor = np.concatenate([water_y_cor, square_y])
            acc=acc+10
        water_points = np.array(list(zip(water_x_cor, water_y_cor)))

        # add cluster and article_id to water points
        self.points = np.concatenate([self.points, water_points])

        water_point_id = len(set(self.cluster_list))
        for index in range(num_points, len(self.points)):
            self.cluster_list = np.append(self.cluster_list, water_point_id)
            self.article_id_list = np.append(self.article_id_list, -1)

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
        vor_points, vor_clusters = [], []

        for index, coor in enumerate(self.points):
            if dominant_cluster[index] == int(self.cluster_list[index]):
                vor_points.append(coor)
                vor_clusters.append(self.cluster_list[index])

        self.points = vor_points
        self.cluster_list = vor_clusters

    def build_graph(self):
        vor = self.vor

        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            center_1 = self.initiate_center(p1, vor.points, self.centers_dic)
            center_2 = self.initiate_center(p2, vor.points, self.centers_dic)

            # add neighboring voronoi polygon
            center_1.add_neighbor(center_2)
            center_2.add_neighbor(center_1)

            # initiate corners
            corner_1 = self.initiate_corner(v1, vor.vertices, self.corners_dic)
            corner_2 = self.initiate_corner(v2, vor.vertices, self.corners_dic)

            # determine if corners are in the water
            if not center_1.is_water or not center_2.is_water:
                corner_1.update_water(False)
                corner_2.update_water(False)

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

            self.centers_dic[p1] = center_1
            self.centers_dic[p2] = center_2
            self.corners_dic[v1] = corner_1
            self.corners_dic[v2] = corner_2
            self.edge_dic[edge_id] = edge

    def assign_elevation(self):
        q = queue.Queue(maxsize=len(self.corners_dic))
        # Traverse all corners, assign is_coast for corner if the corner is on a border
        # These are borders between continents and sea; and borders between cluctsers
        for corner in self.corners_dic.values():
            for edge in corner.protrudes:
                if edge.is_border:
                    corner.update_coast(True)
                    corner.elevation = 0.0
                    q.put(corner)
                break

        # Assign corner elevations following border lines
        while q.qsize() > 0:
            corner = q.get()
            for adjacent_corner in corner.adjacent:
                new_elevation = 0.01 + corner.elevation
                if not corner.is_water and not adjacent_corner.is_water:
                    # elevation of continent should be higher
                    new_elevation += 1 + random.random()
                if new_elevation < adjacent_corner.elevation:
                    # the default elevation is infinity
                    adjacent_corner.elevation = new_elevation
                    q.put(adjacent_corner)

        # Assign center elevations taking the average of corners' elevation
        max_elevation = 0
        for center in self.centers_dic.values():
            sum = 0.0
            for corner in center.corners:
                sum += corner.elevation
            elevation = sum / len(center.corners)
            if max_elevation < elevation:
                max_elevation = elevation

            center.update_elevation(elevation)
        self.max_elevation = max_elevation

    def clockwise_all_corners(self):
        """Clockwise order the corners of a polygon in order to fill color of a bounded polygon"""
        for center in self.centers_dic.values():
            b = True
            for vertex in center.corners:
                if vertex.position[0] < self.bounding_box[0] - 10 or vertex.position[0] > self.bounding_box[1] + 10 or \
                   vertex.position[1] < self.bounding_box[2] - 10 or vertex.position[1] > self.bounding_box[3] + 10:
                    b = False
                    break
            if b:
                center.sort_clockwise()

    def add_noisy_corners(self, p0, p1, v0, v1, min_length=3):
        """Recursively generate new corners based on an original straight border, ordering from v0 to v1"""
        def interpolate(self, pt0, pt1, value=0.5):
            return pt1 + (np.subtract(pt0, pt1) * value)

        points = []
        def subdivide(p0, p1, v0, v1):
            """Randomly find central points within a quadrilateral bounded by p0, p1, v0, v1"""
            if np.linalg.norm(np.subtract(p0, p1)) < min_length or np.linalg.norm(np.subtract(v0, v1)) < min_length:
                return

            rand0, rand1 = np.random.uniform(0.2, 0.8, 2)

            E = interpolate(p0, v0, rand0)
            F = interpolate(v1, p1, rand0)
            G = interpolate(p0, v1, rand1)
            I = interpolate(v0, p1, rand1)

            central_point = interpolate(E, F, rand1)

            subdivide(E, I, v0, central_point)
            points.append(central_point)
            subdivide(G, F, central_point, v1)

        subdivide(p0, p1, v0, v1)
        return np.array(points)     # ordered from v0 to v1. p1 needs to reverse the array to preserve clockwise ordering

    def noise_border(self):
        """
        Noise the border by randomly generating central points within a quadrilateral, store the points as Corners in a path
        Insert the list of corners to the center (forward sequence for p0, reverse sequence for p1 to preserve clockwise ordering)
        """
        for edge in self.edge_dic.values():
            if edge.is_border:
                # the sequence of v0, v1, p0, v1 is important for add the noisy path to the existing polygon
                # v0 is the corner with higher y, the top vertex between the two
                if edge.v0.position[1] > edge.v1.position[1]:
                    v0 = edge.v0
                    v1 = edge.v1
                else:
                    v0 = edge.v1
                    v1 = edge.v0

                # p0 is the center with the lower x, the left center between the two
                if edge.d0.position[0] < edge.d1.position[0]:
                    p0 = edge.d0
                    p1 = edge.d1
                else:
                    p0 = edge.d1
                    p1 = edge.d0

                # the new border path connected with new 2D points as a np.array
                path = self.add_noisy_corners(p0.position, p1.position, v0.position, v1.position)
                if len(path) != 0:
                    # generate new corners and insert to the corresponding center's corners list
                    corner_objects = []
                    for i, position in enumerate(path):
                        id = len(self.corners_dic)
                        corner = Corner(id, position)
                        corner_objects.append(corner)
                        self.corners_dic[id] = corner
                    # insert all corners in the border path
                    p0.insert_corners(v0, corner_objects)
                    p1.insert_corners(v1, np.flip(corner_objects))

    def _create_color(self):
        colors = {}
        num_cluster = len(set(self.cluster_list))-1
        palette = sns.color_palette(self.color_palette, num_cluster).as_hex()
        for i in range(num_cluster):
            colors[i] = sns.light_palette(palette[i], n_colors=int(self.max_elevation)+1).as_hex()
        return colors

    def draw_graph(self):
        ax = plt.subplot()
        patches = []
        colors = self._create_color()

        for center in self.centers_dic.values():
            b = True
            polygon = []
            for vertex in center.corners:
                if vertex.position[0] < self.bounding_box[0] - 10 or vertex.position[0] > self.bounding_box[1] + 10 or \
                   vertex.position[1] < self.bounding_box[2] - 10 or vertex.position[1] > self.bounding_box[3] + 10:
                    if vertex.is_water:
                        b = False
                        break
                polygon.append(vertex.position)

            if b:
                color = '#AADAFF' if center.is_water else colors[center.cluster][int(center.elevation)]
                patches.append(Polygon(polygon, color=color))

        p = PatchCollection(patches, alpha=0.8, match_original=True)
        ax.add_collection(p)
        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.show()

    def export_boundaries(self, directory):
        row_list = []
        for id, edge in self.edge_dic.items():
            if edge.is_border:
                a, b = edge.v0.position, edge.v1.position
                row_list.append({'x1': a[0], 'y1': a[1], 'x2': b[0], 'y2': b[1]})
        pd.DataFrame(row_list).to_csv(directory + "/boundary.csv", index=False)

    def export_polygons(self, directory):
        row_list = []
        colors = self._create_color()

        for center in self.centers_dic.values():
            b = True
            polygon = []
            for vertex in center.corners:
                if vertex.position[0] < self.bounding_box[0] - 10 or vertex.position[0] > self.bounding_box[1] + 10 or \
                   vertex.position[1] < self.bounding_box[2] - 10 or vertex.position[1] > self.bounding_box[3] + 10:
                    if vertex.is_water:
                        b = False
                        break
                polygon.append(vertex.position)

            if b:
                color = '#AADAFF' if center.is_water else colors[center.cluster][int(center.elevation)]
                temp = []
                for vertex in polygon:
                    # vertices coordinates clockwise
                    temp.append(vertex[0])
                    temp.append(vertex[1])

                temp.append(color) # append color at the end
                row_list.append(temp)

        pd.DataFrame.from_records(row_list).to_csv(directory + '/elevation.csv', index=False)
