import pandas as pd
import random


def assign_groups(concat_country, num_per_user):
    group = []
    cluster_groups = []

    remainder = len(concat_country) % 30

    for i in range(len(concat_country) - remainder):
        if len(group) >= 30:
            cluster_groups.append(group)
            group = []
        else:
            group.append(concat_country[i])

    if remainder > 0:
        for i in range(len(concat_country) - remainder, len(concat_country)):
            group.append(concat_country[i])
        sample = concat_country[0:len(concat_country) - remainder]
        while len(group) < num_per_user:
            article = random.choice(sample)
            group.append(article)
            sample.remove(article)
        cluster_groups.append(group)

    return cluster_groups


def concatenate_articles(country, num_concatenations):
    initial = country.copy()
    country_articles = []
    random.shuffle(country)
    country_articles.extend(country)
    for i in range(num_concatenations - 1):
        random.shuffle(initial)
        country_articles.extend(initial)

    cluster_groups = (assign_groups(country_articles, 30))
    return cluster_groups


def get_groups(articles):
    groups = {}

    treatments = {'plain': [[] for x in range(7)], 'augmented': [[] for x in range(7)], 'LDA': [[] for x in range(7)]}

    for row in articles.itertuples():
        treatments[row.treatment][row.country].append(row.article_name)

    for treatment, countries in treatments.items():
        for i in range(len(countries)):
            country_groups = concatenate_articles(countries[i], 5)
            treatments[treatment][i] = country_groups

    for treatment_type, treatment in treatments.items():
        for country in range(len(treatment)):
            for group in range(len(treatment[country])):
                groups[treatment_type + '_c' + str(country) + '_g' + str(group)] = treatment[country][group]

    return groups


def create_articles_df(plain_articles, augmented_articles, LDA_articles):
    plain_augmented = pd.concat([plain_articles, augmented_articles], ignore_index=True)
    articles_df = pd.concat([plain_augmented, LDA_articles], ignore_index=True)
    return articles_df


def main(plain_articles_csv, augmented_articles_csv, LDA_articles_csv):
    plain_articles = pd.read_csv(plain_articles_csv)
    augmented_articles = pd.read_csv(augmented_articles_csv)
    LDA_articles = pd.read_csv(LDA_articles_csv)

    articles_df = create_articles_df(plain_articles, augmented_articles, LDA_articles)
    groups = get_groups(articles_df)
    print(groups)


main('study/food/kmeans_plain/final_articles.csv', 'study/food/kmeans_augmented/final_articles.csv', 'study/food/LDA/final_articles.csv')