import logging
import pandas as pd
import numpy as np

from cartograph.graph.Graph import Graph

def preprocess_file(xy_embedding_csv, cluster_group_csv):
    cluster_list, article_id_list = [], []
    is_match = True

    xy_embedding_df = pd.read_csv(xy_embedding_csv)
    cluster_groups_df = pd.read_csv(cluster_group_csv)

    # check article_ids match on both files
    xy_embedding_id = xy_embedding_df['article_id'].values
    cluster_id = cluster_groups_df['article_id'].values
    for i, j in zip(xy_embedding_id, cluster_id):
        if int(i) != int(j):
            is_match = False
            break

    if is_match:
        cluster_list = cluster_groups_df['country'].values.tolist()
        article_id_list = cluster_groups_df['article_id'].values.tolist()

    else:
        # construct the cluster_list
        logging.warning("Embedding and clustering ids don't match, matching takes extra time")
        for row in xy_embedding_df.itertuples():
            article_id = int(row.article_id)
            cluster = cluster_groups_df.loc[cluster_groups_df['article_id'] == row.article_id, ['country']].iloc[0].tolist()[0]
            cluster_list.append(cluster)
            article_id_list.append(article_id)

    points = np.zeros(shape=(xy_embedding_df.shape[0], 2))
    for index, row in xy_embedding_df.iterrows():
        points[index] = [row['x'], row['y']]

    assert points.shape[1] == 2, 'Required 2D input'

    return points, cluster_list, article_id_list

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    experiment_directory = sys.argv[1]

    points, cluster_list, article_id_list = preprocess_file(experiment_directory + '/xy_embeddings.csv', experiment_directory + '/cluster_groups.csv')
    g = Graph(points, cluster_list, article_id_list)
    g.export_boundaries(experiment_directory)
    g.export_polygons(experiment_directory)
