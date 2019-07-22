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

        self.neighbors = set()  # neighboring Voronoi centers
        self.border = set()    # voronoi bordering edges
        self.corners = set()   # voronoi polygon corners

        self.elevation = 0

    def add_neighbor(self, center):
        self.neighbors.add(center)

    def add_border(self, edge):
        self.border.add(edge)

    def add_corner(self, corner):
        self.corners.add(corner)

    def update_elevation(self, elevation):
        self.elevation = elevation
