"""
    Given domain concepts, output category_names.csv (label id and label string) and
    article_categories.csv (article id and label id)
    Author: Yuren "Rock" Pang and Lily Irvin
"""

import wikipediaapi
from gensim.summarization import keywords
import pandas as pd
import os


def fetch_keywords(domain_concept):
    wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    page = wiki_wiki.page(domain_concept).text
    return keywords(page, lemmatize=True).split('\n')


def create_labels(domain_concept_csv):
    """
    Find the text of each domain concept and creates a data frame with articles and keyword labels
    :return: a dataframe with article id and label id
    """

    # mapping from ids to labels
    labels_to_id = {}
    df = pd.read_csv(domain_concept_csv)
    df = df.iloc[, :]
    rows_list = []
    x = 0
    # Loop through the domain concepts
    for index, row in df.iterrows():  # test df.head(10).iterrows()
        if x % 1000 == 0:
            print(str(x) + ' articles completed')
        article_id = row[0]
        domain_concept = row[1]
        for keyword in fetch_keywords(domain_concept):
            if keyword not in labels_to_id:
                labels_to_id[keyword] = len(labels_to_id)
            id = labels_to_id.get(keyword, len(labels_to_id))
            rows_list.append({"article_id": article_id, "label_id": id})
        x += 1
    return labels_to_id, pd.DataFrame(rows_list)


def create_label_id_str_csv(directory, labels_to_ids):
    id_to_label = [ (id, label) for (label, id) in labels_to_ids.items() ]
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/keyword_names.csv', index=False)


def create_article_label_csv(article_label_df, directory):
    article_label_df.to_csv(directory + "/article_keywords.csv", index=False)


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