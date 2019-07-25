"""
Given vectors for domain concepts, produces (x, y) embeddings using UMAP.

Author: Jonathan Scott
"""
import pandas as pd

import os
import argparse
import umap
import warnings
warnings.filterwarnings('ignore')


def create_embeddings(vector_directory, spread, tw, clusters):
    df = pd.read_csv(vector_directory)
    if clusters is not None:
        cluster_groups = pd.read_csv(clusters)
        df = pd.merge(df, cluster_groups, on='article_id')
        points = umap.UMAP(metric='cosine', spread=spread, target_weight=tw).fit_transform(df.iloc[:, 1:-1], y=df.iloc[:, -1])
    else:
        points = umap.UMAP(metric='cosine', spread=spread).fit_transform(df.iloc[:, 1:-1])
    xy_embedding = pd.DataFrame({'article_id': df['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
    return xy_embedding


def create_csv(xy_embeddings, map_directory):
    df = pd.DataFrame(xy_embeddings, columns=['article_id', 'x', 'y'])
    df.to_csv(map_directory + '/xy_embeddings.csv', index=False)


def main(map_directory, vector_directory, spread, tw, clusters):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    xy_embedding = create_embeddings(vector_directory, spread, tw, clusters)
    create_csv(xy_embedding, map_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce dimensionality of vectors to 2D using UMAP.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--spread', default=20.0)
    parser.add_argument('--clusters', default=None)
    parser.add_argument('--tw', default=0.5)

    args = parser.parse_args()
    main(args.experiment, args.vectors, float(args.spread), float(args.tw), args.clusters)


