from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from math import sqrt

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def draw_regre(X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    model = LinearRegression()
    reg = model.fit(X.reshape(-1, 1), Y)

    pred = model.predict(X_test.reshape(-1, 1))
    rms = sqrt(mean_absolute_error(y_test, pred))


    ## spline
    weights = np.polyfit(X_train, y_train)
    model = np.poly1d(weights)
    pred = model(X_test)

    xp = np.linspace(X_test.min(), X_test.max(), 70)
    pred_plot = model(xp)
    plt.scatter(X_test, y_test, facecolor='None', edgecolor='k', alpha=0.3)
    plt.plot(xp, pred_plot)
    plt.show()

    # this is linear regression
    return rms


def main(hit_labels_csv, tfidf_csv):
    hit_labels_df = pd.read_csv(hit_labels_csv)
    tfidf_df = pd.read_csv(tfidf_csv)

    avg_borda = hit_labels_df[["name", "avg_borda"]]
    tfidf = tfidf_df[["name", "tfidf"]]  # change "name" to the correct label

    df = pd.merge(avg_borda, tfidf, left_on="name", right_on="name")
    Y = df["avg_borda"]
    X = df.drop(columns=['avg_borda'])

    draw_regre(X, Y)
