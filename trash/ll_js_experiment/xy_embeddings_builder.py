"""
Given vectors for domain concepts, produces (x, y) embeddings.

Author: Lily Irvin
"""
import pandas as pd
from sklearn.manifold import TSNE
import umap

# how to find a spread? and min_dist? figure out parameters for UMAP
def create_embeddings(csv, dr_method='tsne'):
    df = pd.read_csv(csv)
    clusters = pd.read_csv('data/food/cluster_groups.csv')
    df = pd.merge(df, clusters, on='article_id')
    points = TSNE().fit_transform(df.iloc[:, 2:-1]) if dr_method == 'tsne' else umap.UMAP(metric='cosine',
                                                                                        spread=10.0).fit_transform(df.iloc[:, 2:-1], y=df.iloc[:, -1])
    xy_embedding = pd.DataFrame({'article_id': df['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
    return xy_embedding


def create_csv(xy_embeddings, directory):
    df = pd.DataFrame(xy_embeddings, columns=['article_id', 'x', 'y'])
    df.to_csv(directory + '/xy_embeddings.csv', index=False)


def main(map_directory, dr_method):
    xy_embedding = create_embeddings(map_directory +'/article_vectors.csv', dr_method=dr_method)
    create_csv(xy_embedding, map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory dr_method (tsne or umap)' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    dr_method = sys.argv[2]
    main(map_directory, dr_method)
