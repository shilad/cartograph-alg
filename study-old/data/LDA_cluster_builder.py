import pandas as pd


def create_cluster_df(article_topic_distribution_df):
    article_id = 0
    cluster_groups = {}
    country_scores = []

    for row in article_topic_distribution_df.itertuples():
        if article_id == row.article_id:
            country_scores.append((row.country, row.prob))
        else:
            country_scores.sort(key=lambda tup: tup[1], reverse=True)
            cluster_groups[row.article_id - 1] = country_scores[0][0]
            article_id += 1
            country_scores.clear()
            country_scores.append((row.country, row.prob))

    cluster_groups_df = pd.DataFrame.from_dict(cluster_groups, orient='index', columns=['country'])
    cluster_groups_df['article_id'] = cluster_groups_df.index
    cluster_groups_df = cluster_groups_df.set_index('article_id')
    return cluster_groups_df


def main(directory):
    article_topic_distribution_df = pd.read_csv(directory + '/article_topic_distribution.csv')
    country_clusters = create_cluster_df(article_topic_distribution_df)
    country_clusters.to_csv(directory + 'cluster_groups.csv')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory = sys.argv[1]
    main(directory)
