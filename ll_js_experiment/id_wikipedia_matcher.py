"""
    Given domain concepts, output wiki_ids which matches id in the input domain file and wikipedia id

    Author: Yuren "Rock" Pang
"""

import requests
import logging
import pandas as pd
import os
import time


def get_id_from_dump(domain_concept, df):
    domain_concept = domain_concept.replace(' ', "_")
    id = df.loc[df['page_title'] == domain_concept, 'page_id'].iloc[0]
    return id

def fetch_id_from_wiki(domain_concept):
    """
    GET request to fetch domain concept unique id from wikipedia
    :param domain_concept:
    :return:
    """
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": domain_concept
    }

    R = S.get(url=URL, params=PARAMS)
    data = R.json()
    page_ids =list(data['query']['pages'].keys())
    id = -1

    if len(page_ids) == 1:
        try:
            id = data['query']['pages'][page_ids[0]]['pageid']
        except KeyError:
            logging.warning("Could not find id for %s", domain_concept)
    else:
        if len(page_ids) == 0:
            logging.warning("Could not find id for %s", domain_concept)
        else:
            logging.warning("Could not determine id for %s. Discovered %d pages when expected 1",
                        domain_concept, len(page_ids))

    return id


def match_ids_with_wiki(domain_concept_csv):
    id_to_wiki_id = {}
    row_list = []

    print(domain_concept_csv)
    df = pd.read_csv(domain_concept_csv)
    wiki_df = pd.read_csv("../data/enwiki-latest-page-parsed.csv")

    for index, row in df.iterrows():
        article_id = row[0]
        domain_concept = row[1]
        wiki_id = get_id_from_dump(domain_concept, wiki_df)
        if wiki_id == -1:
            continue
        else:
            row_list.append({"article_id": article_id, "wiki_id": wiki_id})

    return pd.DataFrame(row_list)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    start = time.time()

    id_df = match_ids_with_wiki(map_directory + "/domain_concept.csv")
    print("running time: " + str(time.time()-start))
    id_df.to_csv(map_directory + "/wiki_ids_experiment.csv", index=False)


# df = pd.read_csv("../data/page.csv")
# get_id_from_dump("Miao, Arunachal Pradesh", df)
if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
