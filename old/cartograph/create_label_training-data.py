import pandas as pd
import argparse


def main(vectors, clusters, gold_standard):
    clustered_vectors = pd.merge(vectors, clusters[['article_id', 'country']], on=['article_id'])
    print('clustered vectors', clustered_vectors.head())
    labeled_grouped_vectors = pd.merge(clustered_vectors, gold_standard.iloc[:, 0:2], on=['country'])
    print(labeled_grouped_vectors.head())
    return labeled_grouped_vectors.to_csv(args.output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--vectors', required=True)
    parser.add_argument('--cluster_groups', required=True)
    parser.add_argument('--gold_standard', required=True)
    parser.add_argument('--output_file', required=False)

    args = parser.parse_args()
    article_vectors = pd.read_csv(args.vectors)
    cluster_groups = pd.read_csv(args.cluster_groups)
    gold_standard_labels = pd.read_csv(args.gold_standard)
    main(article_vectors, cluster_groups, gold_standard_labels)
