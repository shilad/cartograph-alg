"""
This takes the original wikipedia page and categorylink files and cleans the hidden categories and unnecessary columns
"""

import pandas as pd
import logging


def get_wiki_id(domain_concept, pages_df):
    domain_concept = domain_concept.replace(' ', "_")
    print(domain_concept)
    id = pages_df.loc[pages_df.iloc[:, 1] == domain_concept].iloc[0][0]
    return id


def filter_hidden(cat_links_df, hidden_cat):
    rows_list = []

    for index, row in cat_links_df.iterrows():
        wiki_id = row[0]
        category = row[1]
        if category in hidden_cat:
            continue
        else:
            rows_list.append({"wiki_id": wiki_id, "category": category})

    return pd.DataFrame(rows_list, columns=["wiki_id", "category"])


def add_category_label_in_page_csv(cat_links_df, pages_df):
    for index, row in cat_links_df.iterrows():
        wiki_id = row[0]
        cl_type = row[2]
        if cl_type == "subcat":
            article_name = pages_df.loc[pages_df.iloc[:, 0] == wiki_id].iloc[0][1]






def get_categories_from_csv(wiki_id, cat_link_df):
    try:
        return cat_link_df[cat_link_df['wiki_id'] == wiki_id]['category'].tolist()
    except:
        logging.warning("Couldnt find categories for wiki_id: %d", wiki_id)
        return []


def get_multiple_level_categories(wiki_id, cat_link_df):
    deep_categories = []
    memo = {}
    curr_layer = get_categories_from_csv(wiki_id, cat_link_df)
    temp_layer = []
    deep_categories.extend(curr_layer)

    for level in range(3):
        for concept in curr_layer:
            con


def create_labels(domain_concept_csv, cat_link_df, pages_df):
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
        wiki_id = get_wiki_id(article_id, pages_df)
        domain_concept = row[1]
        categories = get_categories_from_csv(wiki_id, cat_link_df)


        for cat in fetch_categories_from_json(domain_concept):
            if cat not in labels_to_id:
                labels_to_id[cat] = len(labels_to_id)
            id = labels_to_id.get(cat, len(labels_to_id))
            rows_list.append({"article_id": article_id, "label_id": id})


def main(map_directory, page_file, category_file, hidden_category_file):
    # Pre-process, get the pages_df: id --> article name
    # get the cat_links_df: id --> categories
    pages_df = pd.read_csv(page_file,
                           encoding="UTF-8",
                           header=None,
                           skipinitialspace=True,
                           usecols=[0, 2],
                           nrows=100)

    cat_links_df = pd.read_csv(category_file,
                               encoding="UTF-8",
                               header=None,
                               skipinitialspace=True,
                               usecols=[0, 1],
                               nrows=100)

    hidden_cat_df = pd.read_csv(hidden_category_file, encoding="UTF-8")
    hidden_cat = set(hidden_cat_df[hidden_cat_df.columns[0]].to_list())

    cleaned = filter_hidden(cat_links_df, hidden_cat)


    pages_df.to_csv(map_directory + "/cleaned_wiki_pages")
    cleaned.to_csv(map_directory + "/cleaned_wiki_category_links")

