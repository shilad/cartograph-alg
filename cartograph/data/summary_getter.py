"""
Given the domain concept csv file, return the summary of each domain concept

Author: Yuren "Rock" Pang
"""

import requests
import logging
import pandas as pd
import os
import time


def fetch_extract_from_api(domain_concept):
    """
   GET requrest to fetch links of one domain_concept
   :param domain_concept:
   :return:
   """
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "titles": domain_concept,
        "prop": "extracts",
        "exintro": "true",
        "explaintext": "true",
        "redirects": "1",
    }

    R = S.get(url=URL, params=PARAMS)
    data = R.json()

    # edges cases: if wikitext is empty, doesn't contain *
    #              if wikitext is null
    #              if links is empty

    if 'query' in data:
        query = data['query']
        if 'pages' in query:
            page_ids = list(query['pages'].keys())
            if len(page_ids) == 1:
                page_info = query['pages'][page_ids[0]]
                if 'extract' in page_info:
                    return page_info['extract']    # returns the actual summary extract
                else:
                    logging.warning("%s: found ids, but no extract", domain_concept)
            else:
                logging.warning("%s: has multiple ids", domain_concept)
        else:
            logging.warning("%s: couldn't find any pages, but have query", domain_concept)
    else:
        logging.warning("%s: couldn't be queried", domain_concept)

    return ''


def create_summary_df(domain_concept_csv):
    df = pd.read_csv(domain_concept_csv)
    rows_list = []
    count = 0

    for index, row in df.iterrows():
        if count == 500: print("This is %d th article", count)

        article_id = row[0]
        domain_concept = row[1]
        rows_list.append({"article_id": article_id,
                          "article_name": domain_concept,
                          "extract": fetch_extract_from_api(domain_concept)})
        count = count + 1
    return pd.DataFrame(rows_list)


def main(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    start = time.time()
    summary_df = create_summary_df(directory + "/domain_concept.csv")
    logging.warning("Time Spent: %.3f", time.time() - start)

    summary_df.to_csv(directory + "/article_summary.csv", index=False)

if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
