"""
    This script fetch hierarchical categories of key phrases label candidates
"""

import argparse
import logging, requests
import pandas as pd
from nltk.stem import WordNetLemmatizer
import numpy as np
import pickle, os

memo = {}


def fetch_categories_from_json(domain_concept):
    """
   GET requrest to fetch categories of one domain_concept
   :param domain_concept:
   :return:
   """
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": domain_concept,
        "prop": "categories",
        "clshow": "!hidden",
        "cllimit": "500",
        "redirects": ""     # this is important
    }

    R = S.get(url=URL, params=PARAMS)
    data = R.json()

    categories = []
    page_ids = list(data['query']['pages'].keys())
    if len(page_ids) == 1:
        page_info = data['query']['pages'][page_ids[0]]
        try:
            for cat_info in page_info['categories']:
                title = cat_info["title"]
                # Remove categories "by ..." and "Types of ..."
                if " by" in title:
                    title = title[:title.index(" by")]
                elif "Types of " in title:
                    title = title[title.index("Types of ") + 9:].capitalize()
                categories.append(title.replace("Category:", ""))
                # categories.append(cat_info["title"].replace("Category:", ""))
        except KeyError or IndexError:
            logging.warning('%s: article found, but no category appears.', page_info["title"])
    else:
        logging.warning('Couldnt find categories for %s. Discovered %d pages when expected 1',
                        domain_concept, len(page_ids))
    return categories


def fetch_multiple_level_categories_from_json(domain_concept):
    deep_categories = []
    curr_layer = fetch_categories_from_json(domain_concept)

    temp_layer = []
    deep_categories.extend(curr_layer)

    # for level in range(1):
    #     for concept in curr_layer:
    #         concept = "Category:" + concept
    #         if concept not in memo:
    #             next_layer = fetch_categories_from_json(concept)
    #             memo[concept] = next_layer
    #         else:
    #             next_layer = memo[concept]
    #         temp_layer.extend(next_layer)
    #         deep_categories.extend(curr_layer)
    #
    #     curr_layer = temp_layer.copy()
    #     temp_layer.clear()
    return list(set(deep_categories))


def get_tfidf(h_cat):
    counts = h_cat.groupby(["country", "new_name"]).size().reset_index(name="country_label_count")
    h_cat = pd.merge(h_cat, counts, on=('country', 'new_name'))

    counts = h_cat.groupby(["new_name"]).size().reset_index(name="label_count")
    h_cat = pd.merge(h_cat, counts, on='new_name')

    counts = h_cat.groupby(["country"]).size().reset_index(name="num_country_labels")
    h_cat = pd.merge(h_cat, counts, on='country')

    h_cat['num_countries'] = h_cat['country'].nunique()
    h_cat['num_articles'] = h_cat['label_name'].nunique()
    return h_cat


def normalize_within_country(h_cat):
    h_cat['min'] = h_cat.groupby(["country"])["tfidf"].transform(min)
    h_cat['max'] = h_cat.groupby(["country"])["tfidf"].transform(max)
    h_cat['normalized_tfidf'] = (h_cat['tfidf'] - h_cat['min']) / (h_cat['max'] - h_cat['min'])

    return h_cat


def sum_tfidf(h_cat):
    sum = h_cat.groupby(["country", "new_name"])['tfidf'].sum().reset_index(name="sum")
    h_cat = pd.merge(h_cat, sum, on=['country', 'new_name'])
    return h_cat


def generate_new_matrix(predicted, isSumInKeyPhrase, cache):
    labels = []
    new_labels = []
    black_list = ["wikipedia", "categories", "redirect", "disambiguation", "categories", "pages including", "main topic articles"]
    lemmatizer = WordNetLemmatizer()  # remove noise

    for row in predicted.itertuples():
        if row.label_name in cache:
            for cat in cache[row.label_name]:
                if cat.lower() == "food": continue
                labels.append(row.label_name)
                new_labels.append(cat)
        else:
            cache[row.label_name] = []
            for cat in fetch_multiple_level_categories_from_json(lemmatizer.lemmatize(row.label_name.title())):
                cat = lemmatizer.lemmatize(cat.lower())
                if True in ((black_word in cat) for black_word in black_list):
                    continue
                cache[row.label_name].append(cat)
                if cat.lower() == "food": continue
                labels.append(row.label_name)
                new_labels.append(cat)

    # dic that matches old labels to new h_cat labels
    dic = pd.DataFrame( {'label_name': labels, 'new_name': new_labels})
    final = pd.merge(predicted, dic, how="outer", on="label_name")
    h_cat = final[['country', 'label_name', 'new_name', 'tfidf']]

    h_cat = get_tfidf(h_cat)
    h_cat = normalize_within_country(h_cat)
    h_cat = sum_tfidf(h_cat)

    for index, row in h_cat.iterrows():
        h_cat.loc[index, 'tf'] = ((row['country_label_count']) / row['num_country_labels'])
        h_cat.loc[index, 'idf'] = np.log((row['num_articles']) / (row['label_count'] * 6))

    if isSumInKeyPhrase:
        h_cat['new_tfidf'] = h_cat['tf'] * h_cat['idf'] * h_cat['sum']
    else:
        h_cat['new_tfidf'] = h_cat['tf'] * h_cat['idf']
    return h_cat, cache


def main(experiment_dir, isSumInKeyPhrase):
    label_source = pd.read_csv(experiment_dir + "/key_phrases_top_labels.csv")
    # h_cat = generate_new_df(predicted)
    cache = {}
    target = './data/h_cat_from_top_labels_one_level.pkl'
    if os.path.getsize(target) <= 0:
        h_cat, dic = generate_new_matrix(label_source, isSumInKeyPhrase, cache)
    else:
        file = open('./data/h_cat_from_top_labels_one_level.pkl', "rb")
        cache = pickle.load(file)
        h_cat, dic = generate_new_matrix(label_source, isSumInKeyPhrase, cache)
        file.close()

    file = open('./data/h_cat_from_top_labels_one_level.pkl', "wb")
    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    h_cat.to_csv(experiment_dir + "/check.csv")
    label_choice = h_cat.loc[h_cat.reset_index().groupby(['country'])['new_tfidf'].idxmax()]

    lst = []
    for index, row in label_choice.iterrows():
        lst.append({"country": row["country"], "label_name": row["new_name"]})
    pd.DataFrame(lst).to_csv(experiment_dir + "/final_labels.csv")

    # h_cat.to_csv(experiment_dir + "/h_cat_labels.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for label picker.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--isSumInKeyPhrase', required=True, type=bool)

    args = parser.parse_args()

    main(args.experiment, args.isSumInKeyPhrase)