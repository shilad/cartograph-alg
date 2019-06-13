"""
    Given domain concepts, output label_ids.csv (label id and label string) and
    article_labels.csv (article id and label id)

    Author: Yuren "Rock" Pang
"""

import logging
import requests
import pandas as pd
import os


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
        "cllimit": "500"
    }

    R = S.get(url=URL, params=PARAMS)
    data = R.json()

    categories = []
    page_ids = list(data['query']['pages'].keys())
    if len(page_ids) == 1:
        page_info = data['query']['pages'][page_ids[0]]
        for cat_info in page_info['categories']:
            categories.append(cat_info['title'].replace("Category:", ""))
    else:
        logging.warning('Couldnt find categories for %s. Discovered %d pages when expected 1',
                        domain_concept, len(page_ids))

    return categories


def create_labels(domain_concept_csv):
    """
    Find the categories of each domain concept and creates a data frame with articles and labels
    :return: a dataframe with article id and label id
    """

    # mapping from ids to labels
    labels_to_id = {}
    df = pd.read_csv(domain_concept_csv)
    rows_list = []

    # Loop through the domain concepts
    for index, row in df.iterrows():  # test df.head(10).iterrows()
        article_id = row[0]
        domain_concept = row[1]
        for cat in fetch_categories_from_json(domain_concept):
            if cat not in labels_to_id:
                labels_to_id[cat] = len(labels_to_id)
            id = labels_to_id.get(cat, len(labels_to_id))
            rows_list.append({"article_id": article_id, "label_id": id})

    return labels_to_id, pd.DataFrame(rows_list)


def create_label_id_str_csv(directory, labels_to_ids):
    id_to_label = [ (id, label) for (label, id) in labels_to_ids.items() ]
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/label_ids.csv', index=False)


def create_article_label_csv(article_label_df, directory):
    article_label_df.to_csv(directory + "/article_labels.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    labels_to_id, label_df = create_labels(map_directory + "/domain_concept.csv")
    create_article_label_csv(label_df, map_directory)
    create_label_id_str_csv(map_directory, labels_to_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)