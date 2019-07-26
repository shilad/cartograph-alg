import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


def main(data):
    X, y = data.iloc[:, 2:-2].values, data[['0']].values
    enc = LabelEncoder()
    y_encoded = enc.fit_transform(y.ravel())#.toarray()

    train_X, test_X, train_y, test_y = train_test_split(X, y_encoded, test_size=0.20, random_state=42)
    # print(y)
    # print(y_encoded)
    # print(train_y)
    classifier = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0, n_jobs=-1)
    classifier.fit(train_X, train_y)

    y_pred = classifier.predict(test_X)
    print(metrics.f1_score(test_y, y_pred, average='weighted'))
    #print(classifier.feature_importances_)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output_file', required=False)
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    main(data)