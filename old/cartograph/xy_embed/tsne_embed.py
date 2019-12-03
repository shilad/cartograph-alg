"""
Given vectors for domain concepts, produces (x, y) embeddings using t-SNE.

Authors: Lily Irvin, Jonathan Scott
"""

import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
import argparse


def create_embeddings(vector_file):
    df = pd.read_csv(vector_file)
    distance_matrix = cosine_distances(df.iloc[:, 1:])
    points = TSNE(metric="precomputed").fit_transform(distance_matrix)
    xy_embedding = pd.DataFrame({'article_id': df['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
    return xy_embedding


def create_csv(xy_embeddings, experiment):
    df = pd.DataFrame(xy_embeddings, columns=['article_id', 'x', 'y'])
    df.to_csv(experiment + '/xy_embeddings.csv', index=False)


def main(experiment, vector_file):
    xy_embedding = create_embeddings(vector_file)
    create_csv(xy_embedding, experiment)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce dimensionality of vectors to 2D using t-SNE.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    args = parser.parse_args()
    main(args.experiment, args.vectors)
