import pandas as pd
import numpy as np
import argparse

from cartograph.border_graph.Graph import Graph

def preprocess_file(xy_embedding_csv, cluster_group_csv):
    xy_embedding_df = pd.read_csv(xy_embedding_csv)
    cluster_groups_df = pd.read_csv(cluster_group_csv)

    # Merge the xy_embedding df and cluster_groups_df
    combined_df = pd.merge(xy_embedding_df, cluster_groups_df, on='article_id')

    points = np.column_stack((combined_df['x'].values, combined_df['y'].values))
    cluster_list = combined_df['country'].values.astype(int)
    article_id_list = combined_df['article_id'].values.astype(int)

    return points, cluster_list, article_id_list


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--xy_embeddings', required=True)
    parser.add_argument('--cluster_groups',required=True)
    args = parser.parse_args()

    xy_embeddings_file = args.experiment + '/' + args.xy_embeddings
    cluster_groups_file = args.experiment + '/' + args.cluster_groups

    points, cluster_list, article_id_list = preprocess_file(xy_embeddings_file, cluster_groups_file)
    g = Graph(points, cluster_list, article_id_list)
    g.export_boundaries(args.experiment)
    g.export_polygons(args.experiment)
