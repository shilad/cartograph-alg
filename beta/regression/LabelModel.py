from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
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
            # self.predictors_seq = ["h_cat_tfidf", "h_cat_pmi", "links_tfidf", "links_pmi", "key_words_tfidf",
            #                        "key_words_pmi", "key_phrases_tfidf", "key_phrases_pmi", "lda_tfidf", "lda_pmi"]
            self.predictors_seq = ["h_cat_tfidf", "key_words_tfidf", "key_phrases_tfidf",  "links_tfidf", "lda_tfidf" ]
            self.predictors = self.df[self.predictors_seq].copy().values
            self.response = self.df[[metric_share_or_avg_borda]].copy().values
            self.model = None
            self.modelType = ""

    def run_linear_regression(self):
        # convert df to 2d np array
        model = LinearRegression()
        model.fit(self.predictors, self.response)
        print(model.coef_)
        self.model = model
        self.modelType = "Linear"

    def run_gradient_boost(self):
        model = GradientBoostingRegressor(random_state=1234)
        model.fit(self.predictors, self.response)
        self.model = model
        self.modelType = "GradientBoost"

    def print_mse(self):
        # 10-fold cross validation, and fit model
        kfold = KFold(n_splits=10)
        score = sys.maxsize
        for train_index, test_index in kfold.split(self.predictors):
            train_x, test_x = self.predictors[train_index], self.predictors[test_index]
            train_y, test_y = self.response[train_index], self.response[test_index]
            if self.modelType is "Linear":
                model = LinearRegression()
            elif self.modelType is not "GradientBoost":
                model = GradientBoostingRegressor()
            else:
                logging.warning("Error: model has not been selected.")
                break
            model.fit(train_x, train_y)
            rms = mean_squared_error(test_y, model.predict(test_x))

            score = min(rms, score)
        return score

    def predict(self, entry_row):
        # the input should be a dictionary (key: column name, value: value)
        # Return the value given predictors
        lst_of_predictors = []
        for col in self.predictors_seq:
            if col not in entry_row.keys():
                logging.warning(col + " not found in this entry row.")
                break

            lst_of_predictors.append(entry_row[col])
        # print(np.array([lst_of_predictors]))

        # print(self.predictors_seq)
        return self.model.predict(np.array([lst_of_predictors]))[0][0]
#
# model = LabelModel("/Users/senresearch/PycharmProjects/cartograph-alg/regression/bug_fixed.csv")
# print(model.model.predict([[0.02032077059020552, 0.16918848316764934, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])[0][0])