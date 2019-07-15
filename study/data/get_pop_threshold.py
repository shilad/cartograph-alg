import pandas as pd

article_names = pd.read_csv('/Users/research/Documents/Projects/cartograph-alg/data/science/domain_concept.csv')
article_pop = pd.read_csv('/Users/research/Documents/Projects/cartograph-alg/data/science/popularity_score.csv')
article_clusters = pd.read_csv('/Users/research/Documents/Projects/cartograph-alg/study/science/kmeans_augmented/cluster_groups.csv')


def main(article_names, article_pop, article_clusters):
    pop_articles = [[] for x in range(7)]
    for row in article_clusters.itertuples():
        if row.article_id in article_pop['popularity_score']:
            # if article_pop['popularity_score'].iloc[row.article_id] > 50000:
                pop_articles[row.country].append((article_names['article_name'].iloc[row.article_id], article_pop['popularity_score'].iloc[row.article_id]))
    return pop_articles


pop_articles = main(article_names, article_pop, article_clusters)
total_articles = 0
for item in pop_articles:
    total_articles += len(item)
    print(len(item))

print('total: ' + str(total_articles))
for item in pop_articles[0]:
    print(item)
