import pandas as pd


def get_popular_articles(article_info, threshold):
    pop_articles = [[] for x in range(7)]
    for row in article_info.itertuples():
        if row.popularity_score > threshold:
            pop_articles[row.country].append(row.article_name)

    for i in range(len(pop_articles)):
        if len(pop_articles[i]) < 30:
            country_articles = []
            for row in article_info.itertuples():
                if row.country == i:
                    country_articles.append((row.article_name, row.popularity_score))
            country_articles.sort(key=lambda x: x[1], reverse=True)
            for article in country_articles:
                if len(pop_articles[i]) == 30:
                    break
                else:
                    if article not in pop_articles[i]:
                        pop_articles[i].append(article)

    return pop_articles


def main(article_names, article_pop, article_clusters, treatment, directory, threshold):
    names_pop = pd.merge(article_names, article_pop, on='article_id')
    article_info = pd.merge(names_pop, article_clusters, on='article_id')

    pop_articles = get_popular_articles(article_info, int(threshold))

    index = 0
    final_articles_df = pd.DataFrame(columns=['article_name', 'country', 'treatment'])
    for i in range(len(pop_articles)):
        for x in range(len(pop_articles[i])):
            final_articles_df.loc[index] = [pop_articles[i][x], i, treatment]
            index += 1

    final_articles_df.to_csv(directory + '/final_articles.csv')

    # for item in pop_articles:
    #     for article in item:
    #         print(article)
    #
    # total_articles = 0
    # for item in pop_articles:
    #     total_articles += len(item)
    #     print(len(item))
    #
    # print('total: ' + str(total_articles))


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 5:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory, map_data, treatment, threshold = sys.argv[1:]

    article_names = pd.read_csv(map_data + '/domain_concept.csv')
    article_pop = pd.read_csv(map_data + '/popularity_score.csv')
    article_clusters = pd.read_csv(directory + '/cluster_groups.csv')

    main(article_names, article_pop, article_clusters, treatment, directory, threshold)

