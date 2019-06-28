import pandas as pd
import os
import time
import logging

# article_id --> label_id (category, links, keyword)
# set of labels assign new id
# for i -- > all article_id
#     id = get_id from one
#     for file --> 3, go find all labels in each class
#         find corresponding id for that row <-- from dictionary to get string
#         find the id based on this string
# output the id file
# output article file

# def create_sublabel_dic(label_candidate_csv):
#     df = pd.read_csv(label_candidate_csv)
#     return df.set_index('label_id')['label'].to_dict()

def label_combiner(article_categories, article_keywords, article_links,
                   categories_names, keyword_names, links_names,
                   domain_concept):
    # dictionary correpondings to id : string of original label candidate
    cat_dic = categories_names.set_index('label_id')['label'].to_dict()
    keyword_dic = keyword_names.set_index('label_id')['label'].to_dict()
    links_dic = links_names.set_index('label_id')['label'].to_dict()

    label_candidates_list = [article_categories, article_keywords, article_links]
    label_candidates_dic = [cat_dic, keyword_dic, links_dic]

    # create complete label candidates
    # total_label_str = list(set(cat_dic.values()) | set(keyword_dic.values()) | set(links_dic.values()))
    # total_label_to_id = dict(map(lambda t: (t[1], t[0]), enumerate(total_label_str)))

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
    links_df.to_csv(directory + '/label_names_complete.csv', index=False)


def create_article_link_csv(article_link_df, directory):
    article_link_df.to_csv(directory + "/article_labels_complete.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    start = time.time()

    domain_concept_df = pd.read_csv(map_directory + '/domain_concept.csv')
    article_categories_df = pd.read_csv(map_directory + '/article_hierarchical_categories.csv')
    article_keywords_df = pd.read_csv(map_directory + '/article_keywords_summary.csv')
    article_links_df = pd.read_csv(map_directory + '/article_links.csv')

    category_names_df = pd.read_csv(map_directory + '/hierarchical_category_names.csv')
    keyword_names_df = pd.read_csv(map_directory + '/keyword_summary_names.csv')
    link_names_df = pd.read_csv(map_directory + '/link_names.csv')

    links_to_id, link_df = label_combiner(article_categories_df, article_keywords_df, article_links_df,
                                          category_names_df, keyword_names_df, link_names_df, domain_concept_df)

    logging.warning("Time Spent: %.3f", time.time() - start)
    create_article_link_csv(link_df, map_directory)
    create_link_id_str_csv(map_directory, links_to_id)


# if __name__ == '__main__':
#     import sys
#
#     if len(sys.argv) != 2:
#         sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
#         sys.exit(1)
#
#     map_directory = sys.argv[1]
#     main(map_directory)



main('../../data/food')



