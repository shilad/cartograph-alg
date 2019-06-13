"""
Given a list of vector representations for all Wikipedia articles,
and a list of Wikipedia titles representing our domain concepts,
output a data frame containing the article id and the vectors.
Author: Lily Irvin, Jonathan Scott
"""

import logging

import pandas as pd
import time


def read_domain_concepts(path):
    return pd.read_csv(path+'/domain_concept.csv')


def vec_str_to_float(string):
    try:
        return float(string)
    except ValueError:
        pass


def read_vectors(vec_path):
    vectors = {}
    start = time.time()
    with open(vec_path, encoding="UTF-8", errors='replace') as file:
        header = file.readline()
        num_dimensions = int(header.split()[1])
        for line_num, line in enumerate(file):
            try:
                tokens = line.split()
                title, values = tokens[0], tokens[1:]
                if num_dimensions < 0:
                    num_dimensions = len(values)
                if len(values) != num_dimensions:
                    logging.warning("invalid vector length in line %d for article %s (%d)",
                                    line_num, title, len(values))
                vectors[title] = [float(x) for x in values]
                if line_num % 100000 == 0:
                    logging.warning("decoding vector %d for article %s", line_num, title)
            except ValueError:
                logging.warning("invalid float in line %d: %s", line_num, repr(line))
            except UnicodeDecodeError:
                logging.warning("invalid encoding in line %d: %s", line_num, repr(line))

    end = time.time()
    # print("Reading in data takes: "+str(end-start)+" seconds.")
    return vectors


def map_domain_names_to_ids(domain_concept_df):
    temp_map_dict = {}  # for fast look up
    for _, row in domain_concept_df.iterrows():
        temp_map_dict[row['article_name'].replace(" ", "_")] = row['article_id']
    return temp_map_dict


def create_domain_vector_csv(orig_vectors, domain_names_to_ids, map_directory):
    some_vector = next(iter(orig_vectors.values()))
    num_dimensions = len(some_vector)
    id_name_pairs = [ (id, name) for (name, id) in domain_names_to_ids.items() ]

    num_missing = 0
    rows = []
    for id, name in sorted(id_name_pairs):
        if name in orig_vectors:
            vector = orig_vectors[name]
            fields = { 'article_id' : id }
            for i in range(num_dimensions):
                fields['vector_' + str(i)] = vector[i]
            rows.append(fields)
        else:
            num_missing += 1
    logging.warning('did not find vectors for %d articles', num_missing)

    columns = ['article_id'] + ['vector_' + str(i) for i in range(num_dimensions)]
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(map_directory + "/article_vectors.csv", index=False)


def main(map_directory, orig_vector_directory):
    domain_concept_df = read_domain_concepts(map_directory)
    vectors = read_vectors(orig_vector_directory)
    domain_names_to_ids = map_domain_names_to_ids(domain_concept_df)
    create_domain_vector_csv(vectors, domain_concept_df, domain_names_to_ids, map_directory)


if __name__ == '__main__':
    read_vectors('data/original_vectors')

    # import sys
    # if len(sys.argv) != 3:
    #     sys.stderr.write('Usage: %s map_directory vector_directory\n' % sys.argv[0])
    #     sys.exit(1)
    #
    # map_directory, orig_vector_directory = sys.argv[1:]
    # main(map_directory, orig_vector_directory)
