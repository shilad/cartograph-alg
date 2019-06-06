"""
    Given domain concepts, output label_ids.csv (label id and label string) and
    article_labels.csv (article id and label id)

    Author: Yuren "Rock" Pang
"""

"""
1. get url
2. get categories through get request
3. Hashset to assign each category with an id
4. key-value categories
5. key-value article and categories
"""
import requests
import pandas as pd
import os

label_id_str_dic = {}
id_counter = 0


def find_categories(json_dic):
    """
    Recursively find categories in the json tree
    :param dict_var:
    :return:
    """
    for k, v in json_dic.items():
        if k == "categories":
            yield v
        elif isinstance(v, dict):
            for id_val in find_categories(v):
                yield id_val


def build_category_list(categories_generator):
    """
    Given a domain concept, build a list of categories from json from API call
    :param categories_generator:
    :return:
    """
    global label_id_str_dic, id_counter
    category_list = []

    # Convert the generator to a list of dictionary
    temp_lst = list(categories_generator)[0]

    for cat in temp_lst:
        # Clean the category string, removing 'Category:'
        curr_category = cat['title'].replace("Category:", "")

        # Update the label_id_str_dic, with new id
        if curr_category not in label_id_str_dic.keys():
            label_id_str_dic[curr_category] = id_counter
            id_counter = id_counter + 1

        category_list.append(curr_category)

    return category_list


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

    # go to the subcategory 'query' -> 'pages'
    categories = find_categories(data["query"]["pages"])

    return build_category_list(categories)


def create_labels(domain_concept_csv):
    """
    Find the categories of each domain concept and prepare for a dataframe construction
    :param domain_concept_csv:
    :return: a dataframe with article id and label id
    """
    global label_id_str_dic

    df = pd.read_csv(domain_concept_csv)
    rows_list = []

    # Loop through the domain concepts
    for index, row in df.iterrows():
        article_id = row[0]
        domain_concept = row[1]
        categories = fetch_categories_from_json(domain_concept)
        # Use key-value pair to prepare a pd df creation
        for cat in categories:
            temp_row_dic = {}
            temp_row_dic.update({"article_id": article_id, "labels": cat})
            rows_list.append(temp_row_dic)

    # Replace the labels with label_id
    for row in rows_list:
        row.update({"label_id": label_id_str_dic[row["labels"]]})

    article_label_df = pd.DataFrame(rows_list)
    article_label_df.drop(columns='labels', inplace=True)

    return article_label_df


def create_label_id_str_csv(directory):
    global label_id_str_dic

    label_id_str_df = pd.DataFrame(list(label_id_str_dic.items()), columns=["label", "label_id"])  # category label_id and label
    label_id_str_df.to_csv(directory + '/label_ids.csv', index=False)


def create_article_label_csv(article_label_df, directory):
    article_label_df.to_csv(directory + "/article_labels.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    label_df = create_labels(map_directory + "/domain_concept.csv")
    create_article_label_csv(label_df, map_directory)
    create_label_id_str_csv(map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
