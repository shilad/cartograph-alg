import logging
import pandas as pd
import numpy as np

from cartograph.border_graph.Graph import Graph
from geojson import FeatureCollection, Feature, MultiPolygon, dumps


def preprocess_file(xy_embedding_csv, cluster_group_csv):
    xy_embedding_df = pd.read_csv(xy_embedding_csv)
    cluster_groups_df = pd.read_csv(cluster_group_csv)

    # Merge the xy_embedding df and cluster_groups_df
    combined_df = pd.merge(xy_embedding_df, cluster_groups_df, on='article_id')

    points = np.column_stack((combined_df['x'].values, combined_df['y'].values))
    cluster_list = combined_df['country'].values.astype(int)
    article_id_list = combined_df['article_id'].values.astype(int)

    return points, cluster_list, article_id_list

def read_label(label_csv):
    label_df = pd.read_csv(label_csv)
    dic = {}
    for row in label_df.itertuples():
        dic[int(row.country)] = row.label_name
    return dic

def export_geojson(directory, multipolgon_dic, label_dic):
    feature_list = []
    for cluster, polygon in multipolgon_dic.items():
        properties = {"clusterID": int(cluster), "label": label_dic[int(cluster)]}
        feature_list.append(Feature(geometry=MultiPolygon(polygon), properties=properties))
    collection = FeatureCollection(feature_list)
    with open(directory, "w") as file:
        file.write(dumps(collection))


# if __name__ == '__main__':
#     import sys
#     if len(sys.argv) != 4:
#         sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
#         sys.exit(1)
#
#     experiment_directory, xy_embeddings, cluster_groups = sys.argv[1:]
#
#     points, cluster_list, article_id_list = preprocess_file(experiment_directory + xy_embeddings, experiment_directory + cluster_groups)
#     g = Graph(points, cluster_list, article_id_list)
#     # g.export_boundaries(experiment_directory)
#     # g.export_polygons(experiment_directory)
#     g.export_boundaries(experiment_directory)

experiment_directory, xy_embeddings, cluster_groups = \
    "/home/rock/Desktop/cartograph-alg/experiments/1111", \
    "/new_xy_embeddings.csv", \
    "/key_phrases_cluster_groups.csv"

points, cluster_list, article_id_list = preprocess_file(experiment_directory + xy_embeddings, experiment_directory + cluster_groups)
g = Graph(points, cluster_list, article_id_list)
# g.export_boundaries(experiment_directory)
# g.export_polygons(experiment_directory)
country_multipolygons = g.get_countries()
export_geojson(experiment_directory + "/countries_new_try.geojson", country_multipolygons, {0:"a", 1:"b", 2:"c", 3:"d", 4:"e", 5:"f", 6:"g", 7:"h", 8:"df"})

# g.export_geojson(experiment_directory + "/countries_new_try.geojson")