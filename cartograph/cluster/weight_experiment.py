import pandas as pd
import numpy as np
from cartograph.cluster.joint_cluster_builder import main
from cartograph.evaluation.cluster_validation_metrics import get_silhouette_score
from cartograph.cluster.cluster_generator import KMeans
import cartograph.utils.tfidf_util as utils
from scipy.sparse import csr_matrix
from sklearn.metrics.cluster import adjusted_rand_score
import pandas as pd
import sklearn.metrics as metrics

def create_sparse_label_matrix(article_labels, tf_idf_score):
    """Creates a matrix that contains article ids and label ids,
    the entry of which is the label score from gensim (if available) or tf-idf score."""
    num_row = max(article_labels['article_id']) + 1
    num_col = max(article_labels['label_id']) + 1
    output_matrix = csr_matrix((num_row, num_col), dtype=np.float).toarray()
    # if 'score' not in article_labels.columns:
    article_labels = article_labels.merge(tf_idf_score, on=['article_id', 'label_id'])
    for row in article_labels.itertuples():
        output_matrix[row.article_id][row.label_id] = row.tfidf

    return output_matrix

low_weights = [0.14, 0.15, 0.11]
# low_weights = [0.09]
# label_weights = [0.12, 0.13, 0.14, 0.15]
# label_weights = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15]
experiment_directory = "/home/rockpang/Desktop/cartograph-alg/experiments/food/0001/"
experiment_path = "/home/rockpang/Desktop/cartograph-alg/experiments/food/0001/"
data_path = "/home/rockpang/Desktop/cartograph-alg/data/food/"
vectors = experiment_path + "vanilla_vectors.csv"
xy_embeddings = experiment_path + "original_xy_embeddings.csv"
output_file = experiment_path + "key_phrases_cluster_groups.csv"
output_embedding = experiment_path + "new_xy_embeddings.csv"
articles_to_labels = data_path + "article_keyphrases.csv"
label_names = data_path + "keyphrases_names.csv"

vanilla_vectors = pd.read_csv(vectors)
article_ids = pd.DataFrame(vanilla_vectors['article_id'])
xy_embeddings = pd.read_csv(xy_embeddings)
articles_to_labels_orig = pd.read_csv(articles_to_labels)
label_names = pd.read_csv(label_names)

k = 9

rows = []

for low_weight in low_weights:
    if low_weight is 0.14:
        label_weights = [0.14, 0.15]
    else:
        label_weights = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.13, 0.14, 0.15]
    for label_weight in label_weights:
        print(low_weight, " ", label_weight)
        if low_weight + label_weight > 1.0: break

        km = KMeans(k)
        orig_groups, orig_average_distance, centroids = km.fit_original_kmeans(vanilla_vectors)
        orig_groups = article_ids.join(pd.DataFrame(orig_groups))
        orig_groups.columns = ['article_id', 'country']
        orig_groups.to_csv('%s/orig_cluster_groups.csv' % (experiment_directory,), index=False)
        articles_to_labels = pd.merge(articles_to_labels_orig, label_names, on="label_id")

        tf_idf_score = utils.calc_tfidf(articles_to_labels, orig_groups, ['article_id', 'label_id', 'tfidf'])
        print("Done with tfidf")

        # joint cluster
        sparse_matrix = create_sparse_label_matrix(articles_to_labels, tf_idf_score)  # #article * #labels wide matrix
        filtered_matrix = sparse_matrix[article_ids['article_id'].values]  # only valid articles to cluster

        print("start clustering")
        joint_alg_groups, mean_distance_high, mean_distance_low, mean_distance_label = \
            km.fit_joint_all(vanilla_vectors, orig_groups, article_ids, xy_embeddings, sparse_matrix,
                             filtered_matrix, label_weight, low_weight, output_embedding)
        print("Done with clustering")
        joint_alg_groups = pd.DataFrame(joint_alg_groups)
        joint_alg_groups.columns = ['article_id', 'country']


        # calcualte siluette
        silhouette_score = metrics.cluster.silhouette_score(vanilla_vectors.iloc[:, 1:].values, joint_alg_groups.drop(columns=['article_id']).values.ravel(), metric='cosine')

        # high, low, label average
        rows.append({"low" : low_weight, "label" : label_weight, "high" : 1 - low_weight - label_weight, "silhouette_score" : silhouette_score,
                     "high_dist" : mean_distance_high, "label_dist" : mean_distance_label, "low_dist" : mean_distance_low})
        print(rows)
        print("##############################################")

pd.DataFrame(rows).to_csv("geography-para-results.csv")

