"""
Given the domain concept csv file, return the summary of each domain concept
Author: Yuren "Rock" Pang
"""


import wikipediaapi
import logging
import pandas as pd
import os
import time


wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)


def fetch_text(domain_concept):
    page = wiki_wiki.page(domain_concept).text
    return page


def create_df(domain_concept_df):
    rows_list = []
    count = 0

    for index, row in domain_concept_df.iterrows():
        if count % 1000 == 0:
            print(str(count) + " articles completed")

        article_id = row[0]
        domain_concept = row[1]
        rows_list.append({"article_id": article_id,
                          "article_name": domain_concept,
                          "text": fetch_text(domain_concept)})
        count += 1
    return pd.DataFrame(rows_list)


def main(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    start = time.time()
    df = pd.read_csv(directory + "/domain_concept.csv")
    summary_df = create_df(df)
    logging.warning("Time Spent: %.3f", time.time() - start)

    summary_df.to_csv(directory + "/article_summary.csv", index=False)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
