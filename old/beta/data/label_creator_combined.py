"""
Combine all label candidates (categories, links, keywords)

Author: Yuren 'Rock' Pang
"""

import pandas as pd
import os
import time
import logging


def label_combiner(article_categories, article_keywords, article_links, article_keyphrases,
                   categories_names, keyword_names, links_names, keyphrases_names,
                   domain_concept):
    """
    loop through all the articles in the domain_concept.csv
    Use the id to find corresponding label_id in article_categories, article_keywords, article_links
    Use individual dictionary to get the string of the labels
    Create new row_list and label_to_id in the complete data frame
    """
    # dictionary correpondings to id : string of original label candidate
    cat_dic = categories_names.set_index('label_id')['label'].to_dict()
    keyword_dic = keyword_names.set_index('label_id')['label'].to_dict()
    links_dic = links_names.set_index('label_id')['label'].to_dict()
    keyphrases_dic = keyphrases_names.set_index('label_id')['label'].to_dict()

    label_candidates_list = [article_categories, article_keywords, article_links, article_keyphrases]
    label_candidates_dic = [cat_dic, keyword_dic, links_dic, keyphrases_dic]

    rows_list = []
    label_to_id = {}

    for index, row in domain_concept.iterrows():
        article_id = row[0]

        for df_index, label_candidate_df in enumerate(label_candidates_list):
            # find all label_id corresponding to the article_id as a list
            label_ids_lst = list(label_candidate_df.loc[(label_candidate_df.article_id == article_id), 'label_id'])
            # find the corresponding str in each label candidate dictionary
            label_strs = [label_candidates_dic[df_index][label_id] for label_id in label_ids_lst]

            for label in label_strs:
                if label not in label_to_id:
                    label_to_id[label] = len(label_to_id)
                id = label_to_id.get(label, len(label_to_id))
                rows_list.append({"article_id": article_id, "label_id": id})

    return label_to_id, pd.DataFrame(rows_list)


def create_link_id_str_csv(directory, links_to_ids):
    id_to_link = [(id, link) for (link, id) in links_to_ids.items()]
    links_df = pd.DataFrame(id_to_link, columns=["label_id", "label"])
    links_df.to_csv(directory + '/combined_label_names.csv', index=False)


def create_article_link_csv(article_link_df, directory):
    article_link_df.to_csv(directory + "/article_labels_combined.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    start = time.time()

    domain_concept_df = pd.read_csv(map_directory + '/domain_concept.csv')
    article_h_categories_df = pd.read_csv(map_directory + '/article_hierarchical_categories.csv')
    article_keywords_df = pd.read_csv(map_directory + '/article_keywords.csv')
    article_links_df = pd.read_csv(map_directory + '/article_links.csv')
    article_keyphrases_df = pd.read_csv(map_directory + '/article_keyphrases.csv')

    h_category_names_df = pd.read_csv(map_directory + '/hierarchical_category_names.csv')
    keyword_names_df = pd.read_csv(map_directory + '/keyword_names.csv')
    link_names_df = pd.read_csv(map_directory + '/link_names.csv')
    keyphrases_names_df = pd.read_csv(map_directory + '/keyphrases_names.csv')

    links_to_id, link_df = label_combiner(article_h_categories_df, article_keywords_df, article_links_df, article_keyphrases_df,
                                          h_category_names_df, keyword_names_df, link_names_df, keyphrases_names_df, domain_concept_df)

    logging.warning("Time Spent: %.3f", time.time() - start)
    create_article_link_csv(link_df, map_directory)
    create_link_id_str_csv(map_directory, links_to_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
