from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def run_linear_regression(x, y):
    model = LinearRegression()
    model.fit(x.values, y.values)
    coeff_df = pd.DataFrame(model.coef_, x.columns, columns=['coefficient'])
    print(coeff_df)
    # this is linear regression
    return ""

def run_spline(x, y):



def main(hit_labels_csv):
    data = pd.read_csv(hit_labels_csv)
    df = data[['h_cat', 'key_phrases', 'key_words', 'lda_label', 'links', 'augmented', 'LDA', "plain"]].astype('category')
    df[['tfidf', 'pmi', 'avg_borda']] = data[['tfidf', 'pmi', 'avg_borda']]

    print(df)

    y = df["avg_borda"]
    x = df[['tfidf', 'pmi', 'h_cat', 'key_phrases', 'key_words', 'lda_label', 'links', 'augmented', 'LDA', "plain"]]

    print(x)

    run_linear_regression(x, y)

main("./labels_for_ml_study.csv")