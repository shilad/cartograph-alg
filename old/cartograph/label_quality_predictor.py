"""
This script creates a regression model to predict the 'quality' of the a label.
Author: Jonathan Scott
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn import tree
import matplotlib.pyplot as plt
import graphviz


def sort_label_scores(test_X, test_y, pred_y):
    """
    Sorts the labels and cluster_ids by their scores and predicted scores.
    :param test_X: A Dataframe
    :param test_y: a numpy array
    :param pred_y: a numpy array
    :return:
    """
    # concat true_scores to label_names
    df = test_X.loc[:, ['group_id', 'label']]#pd.DataFrame(test_X[:, 0:2], columns=['group_id', 'label'])
    df['actual_y'] = test_y
    df['pred_y'] = pred_y
    df["pred_rank"] = df.groupby("group_id")["pred_y"].rank("average", ascending=False)
    df["pred_rank_min"] = df.groupby("group_id")["pred_y"].rank("min", ascending=False)
    df["actual_rank"] = df.groupby("group_id")["actual_y"].rank("min", ascending=False)
    #print(df.sort_values(by=['cluster_id', 'pred_y'], ascending=False))
    return df


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
    X_dropped = data.drop(columns=['Unnamed: 0','label','group_id'])
    #print(X_dropped.head())
    feature_cols = X_dropped.columns[:-1]
    #print(feature_cols)
    X, y = data[feature_cols].values, data[['score']].values
    #train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.20, random_state=42)
    #regressor = RandomForestRegressor()
    #regressor = tree.DecisionTreeRegressor(max_depth=len(feature_cols), max_leaf_nodes=12)
    regressor = LinearRegression()
    regressor.fit(X, y)
    print(feature_cols)
    print('R^2:', regressor.score(X, y))
    print(feature_cols)
    print('Lin Reg coeffiecients', regressor.coef_)
    y_pred = regressor.predict(X)
    # dot_data = tree.export_graphviz(regressor,
    #                                 out_file=None,
    #                                 feature_names=feature_cols,
    #                                 filled=True,
    #                                 rounded=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("label_regressor")
    ranks_df = sort_label_scores(data, y, y_pred)
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