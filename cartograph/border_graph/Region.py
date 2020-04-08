import geojson
from shapely.geometry.polygon import Polygon

class Region:
    """
    Region is the polygon without holes inside, the distinct region polygon
    """
    def __init__(self, id, points):
        self.id = id
        self.polygon = Polygon(points)  # points that form a polygon
        self.points = points
        self.is_root = False
        self.holes = []
        self.inner_regions = []

    def is_outside(self, region):
        return self.polygon.contains(region.polygon)

    def add_direct_region(self, region):
        self.holes.append(region)

    def add_region(self, region):
        self.inner_regions.append(region)

    def get_polygon_coordinates(self):
        """
        To use Polygon in geojson, we need to find the inner region
        """
        exterior_points = list(self.polygon.exterior.coords)
        interior_points = []
        for hole in self.holes:
            coords = list(hole.polygon.exterior.coords)
            for id in range(len(coords)):
               coords[id][0] = float(coords[id][0])
               coords[id][1] = float(coords[id][1])
            interior_points.append(coords)

        return geojson.Polygon([
            exterior_points,
            interior_points
        ])['coordinates']

