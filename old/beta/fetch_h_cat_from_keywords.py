import argparse

from sklearn import preprocessing

memo = {}
import logging, requests
import pandas as pd
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1700)


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
        "redirects": ""
    }

    R = S.get(url=URL, params=PARAMS)
    data = R.json()

    categories = []
    page_ids = list(data['query']['pages'].keys())
    if len(page_ids) == 1:
        page_info = data['query']['pages'][page_ids[0]]
        try:
            for cat_info in page_info['categories']:
                categories.append(cat_info['title'].replace("Category:", ""))

        except KeyError or IndexError:
            logging.warning('%s: article found, but no category appears.', page_info["title"])
    else:
        logging.warning('Couldnt find categories for %s. Discovered %d pages when expected 1',
                        domain_concept, len(page_ids))
    # print(categories)
    return categories


def fetch_multiple_level_categories_from_json(domain_concept):
    deep_categories = []
    curr_layer = fetch_categories_from_json(domain_concept)

    temp_layer = []
    deep_categories.extend(curr_layer)

    for level in range(1):
        for concept in curr_layer:
            concept = "Category:" + concept
            if concept not in memo:
                next_layer = fetch_categories_from_json(concept)
                memo[concept] = next_layer
            else:
                next_layer = memo[concept]
            temp_layer.extend(next_layer)
            deep_categories.extend(curr_layer)

        curr_layer = temp_layer.copy()
        temp_layer.clear()
    return list(set(deep_categories))


def get_tfidf(h_cat, num_top_labels):
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
    # h_cat2
    h_cat['min'] = h_cat.groupby(["country"])["tfidf"].transform(min)
    h_cat['max'] = h_cat.groupby(["country"])["tfidf"].transform(max)
    h_cat['normalized_tfidf'] = (h_cat['tfidf'] - h_cat['min']) / (h_cat['max'] - h_cat['min'])

    print(h_cat)
    return h_cat


def generate_new_matrix(predicted, num_top_labels):
    labels = []
    new_labels = []
    for row in predicted.itertuples():
        for cat in fetch_multiple_level_categories_from_json(lemmatizer.lemmatize(row.label_name.title())):
            cat = lemmatizer.lemmatize(cat.lower())
            if 'wikipedia' in cat or 'categories' in cat or 'type' in cat or 'redirect' in cat or 'disambiguation' in cat:
                continue
            labels.append(row.label_name)
            new_labels.append(cat)

    old_new = pd.DataFrame(
        {'label_name': labels,
         'new_name': new_labels})

    final = pd.merge(predicted, old_new, how="outer", on="label_name")
    h_cat = final[['country', 'label_name', 'new_name', 'tfidf']]
    #h_cat = h_cat.drop_duplicates()
    h_cat = get_tfidf(h_cat, num_top_labels)
    h_cat = normalize_within_country(h_cat)

    for index, row in h_cat.iterrows():
        h_cat.loc[index, 'tf'] = ((row['country_label_count']) / row['num_country_labels'])
        h_cat.loc[index, 'idf'] = np.log((row['num_articles']) / (row['label_count'] * 6))


    h_cat['new_tfidf'] = h_cat['tf'] * h_cat['idf'] * h_cat['normalized_tfidf']
    print(h_cat)
    return h_cat

def main(experiment_dir, num_top_labels):
    predicted = pd.read_csv(experiment_dir + "/key_phrases_top_labels.csv")
    # h_cat = generate_new_df(predicted)
    h_cat = generate_new_matrix(predicted, num_top_labels)
    h_cat.to_csv(experiment_dir + "/h_cat_labels.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for label picker.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--num_top_labels', required=True, type=int)
    args = parser.parse_args()

    main(args.experiment, args.num_top_labels)
