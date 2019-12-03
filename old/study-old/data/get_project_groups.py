import pandas as pd
import random


FOOD = ['Naruto', 'Gmail', 'Urdu', 'Mathematical Statistics', 'Computer Science', 'Blush', 'Painting', 'Earbuds', 'Braces', 'Hairstyle']
MEDIA = ['Tamarind', 'Diapers', 'Baby Powder', 'Lmao', 'Satellite', 'Quiz', 'Vanilla', 'Mistake', 'Four-leaf clover', 'Mac n\' Cheetos', 'Bleach']
INTERNET = ['Aroma of Tacoma', 'Cowboy', 'Birthday Cake', 'The Moon is made of Green Cheese', 'Vampire', '1896 Summer Olympics', 'Caribbean', 'Beach', 'Ramen', 'Braces', 'Chocolate']
TECHNOLOGY = ['American Revolutionary War', 'Serum', 'Old Town Road', 'Sailor Moon', 'Limbo', 'The Lion King', 'Braces', 'Necklace', 'Abdomen', 'Bumblebee']


def create_articles_df(plain_articles, augmented_articles, LDA_articles):
    plain_augmented = pd.concat([plain_articles, augmented_articles], ignore_index=True)
    articles_df = pd.concat([plain_augmented, LDA_articles], ignore_index=True)
    return articles_df


def assign_groups(concat_country, num_per_user, cluster_num, project):
    group = []
    cluster_groups = []

    remainder = len(concat_country) % num_per_user

    for i in range(len(concat_country) - remainder):
        if len(group) == 0:
            if project == 'food':
                group.append(random.choice(FOOD))
            elif project == 'internet':
                group.append(random.choice(INTERNET))
            elif project == 'media':
                group.append(random.choice(MEDIA))
            elif project == 'technology':
                group.append(random.choice(TECHNOLOGY))
        if len(group) >= num_per_user:
            random.shuffle(group)
            group.insert(0, cluster_num)
            cluster_groups.append(group)
            group = []
        else:
            group.append(concat_country[i])

    if remainder > 0:
        if len(group) == 0:
            group.append(cluster_num)
        for i in range(len(concat_country) - remainder, len(concat_country)):
            group.append(concat_country[i])
        sample = concat_country[0:len(concat_country) - remainder]
        while len(group) < num_per_user:
            article = random.choice(sample)
            group.append(article)
            sample.remove(article)
        cluster_groups.append(group)

    return cluster_groups


def concatenate_articles(country, num_concatenations, cluster_num, project):
    initial = country.copy()
    country_articles = []
    random.shuffle(country)
    country_articles.extend(country)
    for i in range(num_concatenations - 1):
        random.shuffle(initial)
        country_articles.extend(initial)

    cluster_groups = (assign_groups(country_articles, 31, cluster_num, project))
    return cluster_groups


def get_groups(articles, project):
    groups = {}

    treatments = {'kmeans_plain': [[] for x in range(7)], 'kmeans_augmented': [[] for x in range(7)], 'LDA': [[] for x in range(7)]}

    for row in articles.itertuples():
        treatments[row.treatment][row.country].append(row.article_name)

    for treatment, countries in treatments.items():
        cluster_num = 0
        for i in range(len(countries)):
            country_groups = concatenate_articles(countries[i], 5, cluster_num, project)
            treatments[treatment][i] = country_groups
            cluster_num += 1

    for treatment_type, treatment in treatments.items():
        for country in range(len(treatment)):
            for group in range(len(treatment[country])):
                groups[treatment_type + '_c' + str(country) + '_g' + str(group)] = treatment[country][group]

    return groups


def create_label_dict(df):
    df_dict = df.to_dict(orient='index')
    for key, value in df_dict.items():
        value.pop('Unnamed: 0', None)
    return df_dict


def create_groups_df(groups, directory, project):
    print(groups)
    article_columns = ['country']
    for i in range(31):
        article_columns.append(project + '_article_' + str(i))

    groups_df = pd.DataFrame.from_dict(groups, orient='index', columns=article_columns)
    group_ids = groups_df.index
    groups_df.insert(0, project + '_group_id', group_ids)
    groups_df.insert(2, 'project', project)

    kmeans_plain_labels = pd.read_csv(directory + '/kmeans_plain/label_candidates.csv')
    plain_dict = create_label_dict(kmeans_plain_labels)
    kmeans_augmented_labels = pd.read_csv(directory + '/kmeans_augmented/label_candidates.csv')
    augmented_dict = create_label_dict(kmeans_augmented_labels)
    LDA_labels = pd.read_csv(directory + '/LDA/label_candidates.csv')
    LDA_dict = create_label_dict(LDA_labels)

    labels = {}
    for row in groups_df.itertuples():
        label_list = []
        if 'kmeans_plain' in row[0]:
            for key, value in plain_dict[row.country].items():
                if isinstance(value, str):
                    label_list.append(value.lower())
                else:
                    label_list.append(value)
        elif 'kmeans_augmented' in row[0]:
            for key, value in augmented_dict[row.country].items():
                if isinstance(value, str):
                    label_list.append(value.lower())
                else:
                    label_list.append(value)
        elif 'LDA' in row[0]:
            for key, value in LDA_dict[row.country].items():
                if isinstance(value, str):
                    label_list.append(value.lower())
                else:
                    label_list.append(value)
        random.shuffle(label_list)
        labels[row.Index] = label_list

    label_columns = []
    for i in range(25):
        label_columns.append(project + '_label_' + str(i))

    labels_df = pd.DataFrame.from_dict(labels, orient='index', columns=label_columns)

    final_groups = pd.merge(groups_df, labels_df, left_index=True, right_index=True)
    return final_groups


def main(plain_articles, augmented_articles, LDA_articles, directory, project):
    articles_df = create_articles_df(plain_articles, augmented_articles, LDA_articles)
    groups = get_groups(articles_df, project)

    groups_df = create_groups_df(groups, directory, project)
    groups_df = groups_df.sample(frac=1)
    groups_df.to_csv(directory + '/groups.csv', index=False)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory, project = sys.argv[1:]

    plain_articles = pd.read_csv(directory + '/kmeans_plain/final_articles.csv')
    augmented_articles = pd.read_csv(directory + '/kmeans_augmented/final_articles.csv')
    LDA_articles = pd.read_csv(directory + '/LDA/final_articles.csv')

    main(plain_articles, augmented_articles, LDA_articles, directory, project)
