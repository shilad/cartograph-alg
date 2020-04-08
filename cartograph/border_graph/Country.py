class Country:
    def __init__(self, cluster):
        self.cluster = cluster
        self.regions = [] # list of region in this country (color)

    def add(self, region):
        self.regions.append(region)