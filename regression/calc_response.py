from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
import sys


def run_linear_regression(predictors, response):
    # convert df to 2d np array
    predictors = predictors.values
    response = response.values

    # 10-fold cross validation
    kfold = KFold(n_splits=10)
    score = sys.maxsize
    ret_model = None
    for train_index, test_index in kfold.split(predictors):
        train_x, test_x = predictors[train_index], predictors[test_index]
        train_y, test_y = response[train_index], response[test_index]
        model = LinearRegression()
        model.fit(train_x, train_y)

        rms = sqrt(mean_squared_error(test_y, model.predict(test_x)))
        if rms < score:
            ret_model = model

    if ret_model is None:
        logging.warning("Error: model has not been selected.")
    return ret_model

def main(hit_label_csv, input_csv):
    df = pd.read_csv(hit_label_csv).iloc[:, 1:]
    predictors = df[["h_cat_tfidf", "h_cat_pmi", "links_tfidf", "links_pmi", "key_words_tfidf", "key_words_pmi",
            "key_phrases_tfidf", "key_phrases_pmi", "lda_tfidf", "lda_pmi"]].copy()

    response_share = df[["share"]].copy()
    # response_avg_borda = df[["avg_borda"]].copy()

    model = run_linear_regression(predictors, response_share)
    print(model.coef_)


    # run_linear_regression(predictors, response_avg_borda)
    #
    # input_df = pd.read_csv(input_csv)
    # lst = input_df.values.tolist()
    # for row in lst:
    #     response = model.predict(row)
    #     row.append(response)
    #
    # ret_df = pd.DataFrame(lst, columns=input_df.columns.tolist().append("avg_borda"))
    # print(ret_df)

main("flattened_data.csv", "")
