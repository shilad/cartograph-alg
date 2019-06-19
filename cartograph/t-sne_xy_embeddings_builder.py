"""
Given vectors for domain concepts, produces (x, y) embeddings using t-SNE.

Authors: Lily Irvin, Jonathan Scott
"""
import pandas as pd
from sklearn.manifold import TSNE
import argparse
import os


def create_embeddings(map_directory, vector_directory):
    df = pd.read_csv(vector_directory)
    points = TSNE().fit_transform(df.iloc[:, 1:])
    xy_embedding = pd.DataFrame({'article_id': df['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
    return xy_embedding


def create_csv(xy_embeddings, map_directory):
    df = pd.DataFrame(xy_embeddings, columns=['article_id', 'x', 'y'])
    df.to_csv(map_directory + '/xy_embeddings.csv', index=False)


def main(map_directory, vector_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    xy_embedding = create_embeddings(map_directory, vector_directory)
    create_csv(xy_embedding, map_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reduce dimensionality of vectors to 2D using t-SNE.')
    parser.add_argument('--map_directory', required=True)
    parser.add_argument('--vectors', required=True)
    args = parser.parse_args()
    main(args.map_directory, args.vectors)
