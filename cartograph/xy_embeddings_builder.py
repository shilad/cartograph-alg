"""
Given vectors for domain concepts, produces (x, y) embeddings.
Author: Lily Irvin
"""
import pandas as pd
from sklearn.manifold import TSNE


def create_embeddings(csv):
    df = pd.read_csv(csv)
    points = TSNE().fit_transform(df.iloc[:, 2:])
    xy_embedding = pd.DataFrame({'article_id': df['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
    return xy_embedding


def create_csv(xy_embeddings, directory):
    df = pd.DataFrame(xy_embeddings, columns=['article_id', 'x', 'y'])
    df.to_csv(directory + '/xy_embeddings.csv', index=False)


def main(map_directory):
    xy_embedding = create_embeddings(map_directory + '/article_vectors.csv')
    create_csv(xy_embedding, map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
