import pandas as pd
import shapely.geometry as geo

def main(experiment):
    geojson = pd.read_json(experiment + "/countries.geojson")
    centroids = []
    num_countries = len(geojson["features"])
    for i in range(num_countries):
        cluster_id = geojson["features"][i]["properties"]["clusterID"]
        cur_country_polygons = geojson["features"][i]["geometry"]["coordinates"]
        cur_country_max_polygon = len(cur_country_polygons[0][0])
        centroid = geo.Polygon(cur_country_polygons[0][0]).centroid
        for j in range(len(cur_country_polygons)):
            cur_polygon_length = len(cur_country_polygons[j][0])
            if cur_polygon_length > cur_country_max_polygon:
                cur_country_max_polygon = cur_polygon_length
                centroid = geo.Polygon(cur_country_polygons[j][0]).centroid
        centroids.append([cluster_id, centroid.x, centroid.y])
    df = pd.DataFrame(centroids, columns=["country", "x", "y"])
    df = df.sort_values(by=["country"])
    df.to_csv(experiment + "/country_centroid.csv", index=False)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='path to the country_centroid.csv file to place labels/auto pan countries in user study')
    parser.add_argument('--experiment', required=True)

    args = parser.parse_args()

    main(args.experiment)
