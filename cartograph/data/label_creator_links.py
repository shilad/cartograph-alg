import requests
import logging
import re
import pandas as pd
import os
import time


def fetch_links_from_api(domain_concept):
    """
   GET requrest to fetch links of one domain_concept
   :param domain_concept:
   :return:
   """
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "parse",
        "format": "json",
        "page": domain_concept,
        "prop": "wikitext",
        "section": "0"
    }

    R = S.get(url=URL, params=PARAMS)
    data = R.json()

    # edges cases: if wikitext is empty, doesn't contain *
    #              if wikitext is null
    #              if links is empty

    if 'parse' in data:
        parse = data['parse']

        if 'wikitext' in parse:
            wikitext_versions = len(parse['wikitext'])
            if wikitext_versions == 1:
                wiki_text = parse['wikitext']['*'][parse['wikitext']['*'].find("'''"):]
                return re.findall("(?<=\[\[)(.+?)(?=[|(\]\])])", wiki_text)
            else:
                logging.warning('%s: found multiple wikitext.', domain_concept)
        else:
            if 'title' in parse:
                logging.warning("%s: Couldn't find wikitext. ", domain_concept)
            else:
                logging.warning("%s: couldn't find anything. ", domain_concept)
    else:
        logging.warning("%s: couldn't be parsed", domain_concept)

    return []


def create_links(domain_concept_csv):
    """
    Find the links of each domain concept and creates a data frame with articles and labels
    :return: a dataframe with article id and label id
    """
    # mapping from ids to labels
    links_to_id = {}
    df = pd.read_csv(domain_concept_csv)
    rows_list = []
    count = 0

    for index, row in df.iterrows():  # test df.head(10).iterrows()
        if count == 500: print("This is %d th article", count)
        article_id = row[0]
        domain_concept = row[1]
        for link in fetch_links_from_api(domain_concept):
            if link not in links_to_id:
                links_to_id[link] = len(links_to_id)
            id = links_to_id.get(link, len(links_to_id))
            rows_list.append({"article_id": article_id, "label_id": id})
        count = count+1

    return links_to_id, pd.DataFrame(rows_list)


def create_link_id_str_csv(directory, links_to_ids):
    id_to_link = [(id, link) for (link, id) in links_to_ids.items()]
    links_df = pd.DataFrame(id_to_link, columns=["label_id", "label"])
    links_df.to_csv(directory + '/link_names.csv', index=False)


def create_article_link_csv(article_link_df, directory):
    article_link_df.to_csv(directory + "/article_links.csv", index=False)


# def main(map_directory):
#     if not os.path.exists(map_directory):
#         os.makedirs(map_directory)
#
#     start = time.time()
#     links_to_id, link_df = create_links(map_directory + "/domain_concept.csv")
#     logging.warning("Time Spent: %.3f", time.time() - start)
#     create_article_link_csv(link_df, map_directory)
#     create_link_id_str_csv(map_directory, links_to_id)
#
#
# if __name__ == '__main__':
#     import sys
#
#     if len(sys.argv) != 2:
#         sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
#         sys.exit(1)
#
#     map_directory = sys.argv[1]
#     main(map_directory)

map_directory = "../../data/food"
start = time.time()
links_to_id, link_df = create_links(map_directory + "/domain_concept.csv")
logging.warning("Time Spent: %.3f", time.time() - start)
create_article_link_csv(link_df, map_directory)
create_link_id_str_csv(map_directory, links_to_id)