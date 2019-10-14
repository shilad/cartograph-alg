import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
def generate_country_matrix(groups):
    """Creates a matrix that contains article ids and label ids,
    the entry of which is the label score from gensim (if available) or tf-idf score."""

    num_row = max(groups['country']) + 1
    num_col = max(groups['article_id']) + 1
    output_matrix = csr_matrix((num_row, num_col), dtype=np.float).toarray()

    for row in groups.itertuples():
        output_matrix[row.country][row.article_id] = 1
    # output_matrix = output_matrix.rename(columns= lambda x: "label_id_" + str(x) )
    # print(output_matrix)
    print(output_matrix.shape)
    return output_matrix

cluster = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/experiments/food/0054/original_cluster_groups.csv")
generate_country_matrix(cluster)