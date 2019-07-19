import sys

class Corner:
    """
    Vertices of a Voronoi polygon
    """
    def __init__(self, id, position):
        self.id = id
        self.position = position
        self.is_coast = False
        self.is_water = True
        self.elevation = sys.maxsize

        self.touches = set()    # set of Center(polygon) touching this corner
        self.protrudes = set()  # a set of edges touching the corner
        self.adjacent = set()   # a set of corners connected to this one

    def add_touches(self, center):
        self.touches.add(center)

    def add_protrudes(self, edge):
        self.protrudes.add(edge)

    def add_adjacent(self, corner):
        self.adjacent.add(corner)

    def update_coast(self, is_coast):
        self.is_coast = is_coast

    def update_elevation(self, elevation):
        self.elevation = elevation

    def update_water(self, is_water):
        self.is_water = is_water
