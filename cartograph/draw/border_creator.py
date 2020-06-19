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

   print(points)
   print(cluster_list)
   print(article_id_list)
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
       print(polygon)
       properties = {"clusterID": int(cluster), "label" : label_dic[int(cluster)]}
       feature_list.append(Feature(geometry=MultiPolygon(polygon), properties=properties))
   collection = FeatureCollection(feature_list)
   with open(directory, "w") as file:
       file.write(dumps(collection))

def get_country_name(final_labels):
   final_labels_df = pd.read_csv(final_labels) # might need to add "index_col=None"
   dic = {}
   for row in final_labels_df.itertuples():
       country_id = int(row.country)
       label_name = row.label_name
       dic[country_id] = label_name

   return dic

# if __name__ == '__main__':
#     import sys
#     from pathlib import Path
#
#     if len(sys.argv) != 4:
#         sys.stderr.write('Usage: % map_directory' % sys.argv[0])
#         sys.exit(1)
#
#     # takes in experiment_directory, xy_embeddings, cluster_groups, and final_labels
#     experiment_directory = Path(sys.argv[1])
#     xy_embeddings, cluster_groups, final_labels = sys.argv[2:]
#
#     points, cluster_list, article_id_list = preprocess_file(experiment_directory / xy_embeddings,
#                                                             experiment_directory / cluster_groups)
#     g = Graph(points, cluster_list, article_id_list)
#     country_multipolygons = g.get_countries()
#     country_names_dic = get_country_name(experiment_directory / final_labels)
#
#
#     export_geojson(experiment_directory / "countries_new_try.geojson", country_multipolygons,
#                    country_names_dic)

from pathlib import Path

experiment_directory, xy_embeddings, cluster_groups, final_labels = \
   Path("../../experiments/0015/"), \
   "original_xy_embeddings.csv", \
   "orig_cluster_groups.csv", \
   "final_labels.csv"

points, cluster_list, article_id_list = preprocess_file(experiment_directory / xy_embeddings,
                                                       experiment_directory / cluster_groups)
g = Graph(points, cluster_list, article_id_list)
g.draw_graph()
# g.find_giant_polygon()
# g.export_multipolygon()
# g.create_regions()
country_multipolygons = g.export_multipolygon()
country_names_dic = get_country_name(experiment_directory / final_labels)
# print(country_names_dic)
#
export_geojson(experiment_directory / "countries_why.geojson", country_multipolygons,
              country_names_dic)
