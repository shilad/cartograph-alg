import pandas as pd
from sklearn.manifold import TSNE
import sys


def create_embeddings(csv):
    df = pd.read_csv(csv)
    points = TSNE().fit_transform(df.iloc[:, 2:])
    xy_embedding = pd.DataFrame({'article_id': df['article_id'], 'x': points[:, 0], 'y': points[:, 1]})
    return xy_embedding


def create_csv(xy_embeddings, directory):
    df = pd.DataFrame(xy_embeddings, columns=['article_id', 'x', 'y'])
    df.to_csv(directory + '/xy_embeddings.csv', index=False)


def main():
    xy_embedding = create_embeddings(sys.argv[1] + '/' + sys.argv[2])
    create_csv(xy_embedding, sys.argv[1])


if __name__ == '__main__':
    main()
