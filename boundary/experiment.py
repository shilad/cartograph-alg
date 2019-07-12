import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import Delaunay

import matplotlib.pyplot as plt


# df = np.array([[1,1],[1,2],[1,3],[2,1],[2,3],[3,2]])



# class Vertex:
#     def __init__(self, index, point, isOnCoast):
#         self.index = index  # Index of point, it is a vertex in the Vornoi region
#         self.x = point[0]
#         self.y = point[1]
#         self.isOnCoast = isOnCoast
#         self.regionPoints = set()
#
#     def addRegionPoints(self, points):
#         self.regionPoints.update(points)
#
#
# class Graph:
#     def __init__(self, points=np.array([])):
#         if len(points) == 0:
#             print("Points need to be passed in.")
#             return
#
#         self.vor_centers = points
#         self.voronoi = Voronoi(points)
#         self.delaunay = Delaunay(points)
#         self.filtered_regions = self.filter_regions(self.voronoi)
#         self.graph = self.create_graph(self.voronoi, self.delaunay)
#
#     def filter_regions(self, voronoi):
#         filtered_regions = []
#         for region in voronoi.regions:
#             if -1 not in region and len(region) != 0:
#                 filtered_regions.append(region)
#         return self.filtered_regions
#
#     def create_graph(self, voronoi, delaunay):
#         edges = {}
#         for triangle in delaunay.slices:
#             for i in range(-1, 2):
#                 edges.update({[triangle[i], triangle[i+1]]: triangle[i]})
#             edge_1 = [triangle[0], triangle[1]]
#             edge_2 = [triangle[1], triangle[2]]
#             edge_3 = [triangle[2], triangle[0]]
#
#
#         points_dic = {index: points for index, points in enumerate(delaunay.points)}



#def create_cluster(cluster_groups_csv):




# points = np.zeros(shape=(df.shape[0], 2))
#
#
# for index, row in df.iterrows():
#     points[index] = [row['x'], row['y']]
# #
# plt.plot(points[:, 0], points[:, 1], 'o', color='black')
# # vor = Voronoi(points)
# #
# #
# #
# # print(vor)
# # voronoi_plot_2d(vor)
# # plt.show()
# #
# #
# # def relax_points(times=2):
# #     for i in range(times):
# #         centroids = []
#
#
#
# import numpy as np
# from networkx import *
# from scipy.spatial import Voronoi
# import sys
#
# class Atlas(object):
#     """
#     Creates a voronoi object, relaxes it as needed, and returns it as a layout.
#     Implements Fortune's Algorithm (https://en.wikipedia.org/wiki/Fortune%27s_algorithm) in python.
#     1. Takes in a 2D numpy array (or generates points with given boundaries)
#     2. Creates a tuple of two graphs (with networkx) representing the Delaunay triangulation (https://en.wikipedia.org/wiki/Delaunay_triangulation)
#     3. Relaxes the points through Lloyd's Algorithm (https://en.wikipedia.org/wiki/Lloyd%27s_algorithm)
#     4. Returns a Voronoi diagram (http://www.voronoi.com/wiki/index.php?title=Main_Page)
#     """
#
#     def __init__(self, points = np.array([]), dimensions = (None, None), granularity = None):
#         """
#         Creates the Voronoi object
#         :param points: predefined points
#         :type points: numpy array of shape (w, 2) where w is the number of points [x, y] style, default None
#         :param dimensions: dimensions of the points, from [w, 2] where w is the highest value, this *cannot* be None if points is None
#         :type dimensions: tuple of ints, maximum (x,y) dimensions, default None
#         :param granularity: how many points to create, must be given if dimensions are given
#         :type granularity: int, default None
#         """
#         if len(points) == 0 and dimensions == (None, None):
#             print('You can\'t have both points and dimensions be empty, try passing in some points or dimensions and granularity.')
#             return
#         if len(points) == 0 and dimensions != None and granularity == None:
#             print('Granularity can\'t be none if dimensions are passed in, try passing in a granularity.')
#             return
#         if len(points) != 0:
#             self.points = points
#         else:
#             points = np.random.random((granularity, 2))
#             points = list(map(lambda x: np.array([x[0]*dimensions[0], x[1]*dimensions[1]]), points))
#             self.points = np.array(points)
#         self.bounding_region = [min(self.points[:, 0]), max(self.points[:, 0]), min(self.points[:, 1]), max(self.points[:, 1])]
#
#     def _eu_distance(self, p1, p2):
#         """
#         Calculates the Euclidian distance between two points
#         :param p1: (x,y) position for the first point
#         :type p1: tuple (or list) of floats
#         :param p2: (x,y) position for the second point
#         :type p2: tuple (or list) of floats
#         :return: the euclidian distance
#         :rtype: float
#         """
#         return np.sqrt(np.power(p1[0]-p2[0], 2) + np.power(p1[1]-p2[1], 2))
#
#     def generate_voronoi(self):
#         """
#         Uses scipy.spatial.Voronoi to generate a voronoi diagram.
#         Filters viable regions and stashes them in filtered_regions, see https://stackoverflow.com/questions/28665491/getting-a-bounded-polygon-coordinates-from-voronoi-cells
#         :return: A voronoi diagram based on the points
#         :rtype: scipy.spatial.Voronoi
#         """
#         eps = sys.float_info.epsilon
#         self.vor = Voronoi(self.points)
#         self.filtered_regions = []
#         for region in self.vor.regions:
#             flag = True
#             for index in region:
#                 if index == -1:
#                     flag = False
#                     break
#                 else:
#                     x = self.vor.vertices[index, 0]
#                     y = self.vor.vertices[index, 1]
#                     if not (self.bounding_region[0] - eps <= x and x <= self.bounding_region[1] + eps and
#                             self.bounding_region[2] - eps <= y and y <= self.bounding_region[3] + eps):
#                         flag = False
#                         break
#             if region != [] and flag:
#                 self.filtered_regions.append(region)
#         return self.vor
#
#     def _region_centroid(self, vertices):
#         """
#         Finds the centroid of the voronoi region bounded by given vertices
#         See: https://en.wikipedia.org/wiki/Centroid#Centroid_of_polygon
#         :param vertices: list of vertices that bound the region
#         :type vertices: numpy array of vertices from the scipy.spatial.Voronoi.regions (e.g. vor.vertices[region + [region[0]], :])
#         :return: list of centroids
#         :rtype: np.array of centroids
#         """
#         signed_area = 0
#         C_x = 0
#         C_y = 0
#         for i in range(len(vertices)-1):
#             step = (vertices[i, 0]*vertices[i+1, 1])-(vertices[i+1, 0]*vertices[i, 1])
#             signed_area += step
#             C_x += (vertices[i, 0] + vertices[i+1, 0])*step
#             C_y += (vertices[i, 1] + vertices[i+1, 1])*step
#         signed_area = 1/2*signed_area
#         C_x = (1.0/(6.0*signed_area))*C_x
#         C_y = (1.0/(6.0*signed_area))*C_y
#         return np.array([[C_x, C_y]])
#
#     def relax_points(self, times=1):
#         """
#         Relaxes the points after an initial Voronoi is created to refine the graph.
#         See: https://stackoverflow.com/questions/17637244/voronoi-and-lloyd-relaxation-using-python-scipy
#         :param times: Number of times to relax, default is 1
#         :type times: int
#         :return: the final voronoi diagrama
#         :rtype: scipy.spatial.Voronoi
#         """
#         for i in range(times):
#             centroids = []
#             for region in self.filtered_regions:
#                 vertices = self.vor.vertices[region + [region[0]], :]
#                 centroid = self._region_centroid(vertices)
#                 centroids.append(list(centroid[0, :]))
#             self.points = centroids
#             self.generate_voronoi()
#         return self.vor
#
#
# vor2 = Atlas(points)
# vor = vor2.generate_voronoi()
# v = vor2.relax_points(2)
#
# from scipy.spatial import Delaunay
# tri = Delaunay(points)
#
# # voronoi_plot_2d(v)
# plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
#
# plt.show()

def build_vor(xy_embedding_csv):
    df = pd.read_csv(xy_embedding_csv)
    points = np.zeros(shape=(df.shape[0], 2))
    for index, row in df.iterrows():
        points[index] = [row['x'], row['y']]

    return Voronoi(points), points.tolist()


def build_boundary_dic(vor):
    """
    point index : True (in the boundary) / False (out of the boundary)
    :param vor:
    :return:
    """
    in_out_dic = {i: False for i in range(len(vor.points))}
    point_region = vor.point_region.tolist()

    for i, vor_vertices in enumerate(vor.regions):
        if len(vor_vertices) != 0 and -1 not in vor_vertices:
            point_index = point_region.index(i)
            in_out_dic.update({point_index: True})

    return in_out_dic


def build_cluster_dic(cluster_groups_csv, xy_embedding_csv, points):
    """
    point index : cluster id
    :param cluster_groups_csv:
    :param xy_embedding_csv:
    :param points:
    :return:
    """
    cluster_groups_df = pd.read_csv(cluster_groups_csv)
    xy_embedding_df = pd.read_csv(xy_embedding_csv)
    id_in_points_to_cluster_dic = {}

    for index, xy in enumerate(points):
        article_id = xy_embedding_df.loc[[index]].iloc[0, :].to_list()[0]

        cluster = cluster_groups_df.loc[cluster_groups_df['article_id'] == article_id, ['country']].iloc[0, :].to_list()[0]
        id_in_points_to_cluster_dic.update({index: cluster})

    # cluster_groups_df = pd.read_csv(cluster_groups_csv)
    # xy_embedding_df = pd.read_csv(xy_embedding_csv)
    # cluster_dic = {}
    #
    # for row in cluster_groups_df.itertuples():
    #     country = row.country
    #     xy_row = xy_embedding_df.loc[xy_embedding_df['article_id'] == row.article_id, ['x', 'y']].iloc[0,:].to_list()
    #     cluster_dic.setdefault(country, []).append([xy_row[0], xy_row[1]])
    #
    # id_in_points_to_cluster_dic = {}
    # for k, v in cluster_dic.items():
    #     for xy in v:
    #         id_in_points_to_cluster_dic.update({points.index(xy): k})
    print(type(id_in_points_to_cluster_dic))
    return id_in_points_to_cluster_dic


def draw_border(id_in_points_to_cluster_dic, in_out_dic, vor):
    """
    If in different territory (cluster & boundary), draw different colors
    :param id_in_points_to_cluster_dic:
    :param in_out_dic:
    :param vor:
    :return:
    """
    vertices_in_vor = vor.vertices
    ridges_in_vor_by_vertices = vor.ridge_vertices
    row_list = []

    for vertex_index, ridge_points in enumerate(vor.ridge_points):
        one_end = ridge_points[0]
        other_end = ridge_points[1]

        if in_out_dic[one_end] ^ in_out_dic[other_end] or \
                id_in_points_to_cluster_dic[one_end] ^ id_in_points_to_cluster_dic[other_end]: ## Completed: This needs an or to include clusters
            print(vor.points[ridge_points[0]], vor.points[ridge_points[1]])
            line = ridges_in_vor_by_vertices[vertex_index]
            a, b = vertices_in_vor[line[0]], vertices_in_vor[line[1]]

            row_list.append({'x1': a[0], 'y1': a[1], 'x2': b[0], 'y2':b[1]})
            plt.plot([a[0], b[0]], [a[1], b[1]], 'ro-', marker='o', markersize=1
                     , linewidth=0.5)

    # pd.DataFrame(row_list).to_csv()
    # voronoi_plot_2d(vor)
    plt.xlim(-300, 300)
    plt.ylim(-300, 300)
    #
    plt.show()


# import drawSvg as draw
# d = draw.Drawing(200, 100, origin='center')
#
# d.append(draw.Line(-80, -45, 70, -49, stroke='black', stroke_dasharray='2,1'))
# d.append(draw.Line(95, 49, -90, 40, stroke='black'))
# d.saveSvg('example.svg')
# # plt.plot([1,1], [2,3])
# # plt.show()
# points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
#                    [2, 0], [2, 1], [2, 2]])
# # vor = Voronoi(points)
vor, points = build_vor('../experiments/food/0009/xy_embeddings.csv')
in_out_dic = build_boundary_dic(vor)
id_in_points_to_cluster_dic = build_cluster_dic('../experiments/food/0009/cluster_groups.csv', '../experiments/food/0009/xy_embeddings.csv', points)

color_scheme = ['blue', 'green', 'yellow', 'black', 'pink', 'red', 'cyan', 'orange']
for index, point in enumerate(points):
    region_index = vor.point_region[index]
    region_with_vertices = vor.regions[region_index]
    color = color_scheme[id_in_points_to_cluster_dic[index]]

    polygon = [vor.vertices[i] for i in region_with_vertices]
    plt.fill(*zip(*polygon), color=color)

# plt.xlim(-100, 100)
# plt.ylim(-100, 100)
# plt.show()
#
draw_border(id_in_points_to_cluster_dic, in_out_dic, vor)
#
# #
# # tri = Delaunay(points)
# #
# # print(vor.ridge_points)
# # print("##simplices")
# # print(tri.simplices)
# #
# # print("----------")
# # print(points[tri.simplices])
# #
# #
# # print("##neighbors")
# # print(tri.neighbors)
# #
# # print("###vertex_to_simplex")
# # print(tri.vertex_to_simplex)
# # print(tri.vertex_neighbor_vertices)
# #
# voronoi_plot_2d(vor)
# # plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
# plt.show()
