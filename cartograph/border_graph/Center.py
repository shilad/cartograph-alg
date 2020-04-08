import numpy as np
import math
import logging
class Center:
    """
    Actual data points, center in a voronoi polygon
    """
    def __init__(self, id, position, cluster, article_id, is_water):
        self.id = id
        self.position = position
        self.cluster = cluster
        self.article_id = article_id
        self.is_water = is_water

        #added
        self.is_border = False

        self.neighbors = set()  # neighboring Voronoi centers
        self.border = set()    # voronoi bordering edges
        self.corners = []   # voronoi polygon corners

        self.elevation = 0

    def add_neighbor(self, center):
        self.neighbors.add(center)

    def add_border(self, edge):
        self.border.add(edge)

    def add_corner(self, corner):
        self.corners.append(corner)

    def update_elevation(self, elevation):
        self.elevation = elevation

    def update_is_border(self, is_border):
        self.is_border = is_border

    def sort_clockwise(self):
        points = np.array([np.array(corner.position) for corner in self.corners])
        mean_x = np.mean(points[:, 0])
        mean_y = np.mean(points[:, 1])

        self.corners = sorted(self.corners, key=lambda x: math.atan2(x.position[1] - mean_y, x.position[0] - mean_x), reverse=True)

    def insert_corners(self, start_corner, corners):

        while self.corners[-1].id != start_corner.id:
            self.corners = np.roll(self.corners, 1, axis=0)

        index = -1
        for i, corner in enumerate(self.corners):
            if corner.position[0] == start_corner.position[0] and corner.position[1] == start_corner.position[1]:
                index = i
                break

        if index == -1:
            logging.warning('Corner not found for this center: %d', self.id)


        for corner in corners:
            self.corners = np.append(self.corners, corner)


