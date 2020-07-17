import numpy as np
from scipy import sparse
from numpy import array
import pandas as pd
import cartograph.utils.tfidf_util as utils

tf_idf_score = pd.read_csv("../tfidf.csv")
article_ids = pd.read_csv("../article_ids.csv")
valid_article_indices = np.array(np.isin(tf_idf_score["article_id"].values, article_ids["article_id"].values))
a = []
for i in range(len(valid_article_indices)):
    a.append(not valid_article_indices[i])
print(a)
print(len(a))




I = tf_idf_score["article_id"][valid_article_indices]
print(len(I))
J = tf_idf_score["label_id"][valid_article_indices]
V = tf_idf_score["tfidf"][valid_article_indices]
num_row = 83325
num_col = 86359
output_matrix = sparse.coo_matrix((V, (I, J)), shape=(num_row, num_col)).tocsc()
# output_matrix = output_matrix[output_matrix.getnnz(1)==0]
print((output_matrix[output_matrix.getnnz(1)==0]).shape)
print((output_matrix[output_matrix.getnnz(1)>0]).shape)
cao = output_matrix[:, article_ids["article_id"]]
print(cao.shape)
print(type(cao))
print(output_matrix.shape)

# print(max(article_ids["article_id"].values))
# # print("b")
#
# output_matrix = sparse.csr_matrix((num_row, num_col), dtype=np.float).toarray()
# # article_labels = article_labels.merge(tf_idf_score, on=['article_id', 'label_id'])
# print("a")
# for row in tf_idf_score.itertuples():
#     output_matrix[row.article_id][row.label_id] = row.tfidf
# print("b")

# a = np.array([1, 2, 3, 4, 5])
# b = [1,2]
# va = np.isin(a, b)
# print(a[va])
#
