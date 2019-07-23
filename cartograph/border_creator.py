import logging
import pandas as pd
import numpy as np

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
#
# experiment_directory = '../experiments/culture/0023'
# points, cluster_list, article_id_list = preprocess_file(experiment_directory + '/xy_embeddings.csv', experiment_directory + '/cluster_groups.csv')
# g = Graph(points, cluster_list, article_id_list, color_palette='hls')
# g.draw_graph()
if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    experiment_directory = sys.argv[1]

    points, cluster_list, article_id_list = preprocess_file(experiment_directory + '/xy_embeddings.csv', experiment_directory + '/cluster_groups.csv')
    g = Graph(points, cluster_list, article_id_list, color_palette='hls')
    g.export_boundaries(experiment_directory)
    g.export_polygons(experiment_directory)
