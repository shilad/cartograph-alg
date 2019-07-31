"""
This script creates a regression model to predict the 'quality' of the a label.
Author: Jonathan Scott
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz


def sort_label_scores(test_X, test_y, pred_y):
    # concat true_scores to label_names
    df = pd.DataFrame(test_X[:, 0:2], columns=['cluster_id', 'label'])
    df['actual_y'] = test_y
    df['pred_y'] = pred_y

    df["pred_rank"] = df.groupby("cluster_id")["pred_y"].rank("average", ascending=False)
    df["pred_rank_min"] = df.groupby("cluster_id")["pred_y"].rank("min", ascending=False)
    df["actual_rank"] = df.groupby("cluster_id")["actual_y"].rank("min", ascending=False)
    #print(df.sort_values(by=['cluster_id', 'pred_y'], ascending=False))
    return df

#
# def label_rank_loss(test_y, pred_y):
#     max_quality = 1.0
#     cat_counter = 0
#
#     for idx, row in test_y.iterrows():
#         if row['label'] == pred_y.loc[idx, 'label']:
#
#
#
#     return


def avg_cluster_rank(ranks_df):
    """
    Calculates average rank picked by our algorithm,lower the better.
    """
    return ranks_df[ranks_df['actual_rank'] == 1.0]['pred_rank'].mean()


def hist_min_rank(ranks_df):
    """
    Look at actual_rank 1.0 and then distribution of pred_rank_min
    """
    rank_1s = ranks_df[ranks_df.actual_rank == 1.0]
    return rank_1s.hist(column=['pred_rank_min'])


def main(data):
    feature_cols = ['cluster_id',
                    'label',
                    'tfidf',
                    'pmi',
                    'h_cat',
                    'key_phrases',
                    'key_words',
                    'lda',
                    'links']
    X, y = data[feature_cols].values, data[['score']].values
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
    regressor = tree.DecisionTreeRegressor()
    regressor.fit(X[:, 2:], y)

    y_pred = regressor.predict(X[:, 2:])
    dot_data = tree.export_graphviz(regressor,
                                    out_file=None,
                                    feature_names=feature_cols[2:],
                                    filled=True,
                                    rounded=True)
    graph = graphviz.Source(dot_data)
    graph.render("label_regressor")

    ranks_df = sort_label_scores(X, y, y_pred)
    print('Model Rank Metric:', avg_cluster_rank(ranks_df))
    hist_min_rank(ranks_df)
    plt.show()
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_file', required=False)
    args = parser.parse_args()
    data = pd.read_csv(args.dataset)
    main(data)