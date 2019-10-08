from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from math import sqrt
import pandas as pd
import numpy as np
import logging
import sys


class LabelModel:
    def __init__(self, training_fro_user_study_csv, metric_share_or_avg_borda="avg_borda"):
        if metric_share_or_avg_borda != 'avg_borda' and metric_share_or_avg_borda != 'share':
            logging.warning("Predictor not found")
        else:
            self.df = pd.read_csv(training_fro_user_study_csv).iloc[:, 1:]
            self.predictors_seq = ["h_cat_tfidf", "h_cat_pmi", "links_tfidf", "links_pmi", "key_words_tfidf",
                                   "key_words_pmi", "key_phrases_tfidf", "key_phrases_pmi", "lda_tfidf", "lda_pmi"]
            self.predictors = self.df[self.predictors_seq].copy()
            self.response = self.df[[metric_share_or_avg_borda]].copy()
            self.model = self.run_linear_regression(self.predictors, self.response)

    def run_linear_regression(self, predictors, response):
        # convert df to 2d np array
        predictors = predictors.values
        response = response.values

        # 10-fold cross validation, and fit model
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

    def predict(self, entry_row):
        # the input should be a dictionary (key: column name, value: value)
        # Return the value given predictors
        lst_of_predictors = []
        for col in self.predictors_seq:
            if col not in entry_row.keys():
                logging.warning(col + " not found in this entry row.")
                break

            lst_of_predictors.append(entry_row[col])
        return self.model.predict(np.array([lst_of_predictors]))[0][0]
#
# model = LabelModel("/Users/senresearch/PycharmProjects/cartograph-alg/regression/bug_fixed.csv")
# print(model.model.predict([[0.02032077059020552, 0.16918848316764934, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])[0][0])