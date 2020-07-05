# Transform popularity score from wikipedia to a score that looks good on cartograph
# Author: Rock Pang,
# Reference: https://github.com/shilad/cartograph/blob/develop-simple/cartograph/CalculateZPop.py

import pandas as pd
import numpy as np
import argparse


def log4(x):
    return np.log2(x) / np.log2(4)


def main(experiment_directory, popularity_score_df, new_xy_embeddings_df):
    assert(popularity_score_df.shape[0] != 0) # check if popularity score df is not empty
    new_rows = []
    pop_dic = {}
    for row in popularity_score_df.itertuples():
        pop_dic[row.article_id] = row.popularity_score

    for row in new_xy_embeddings_df.itertuples():
        val = 0
        if row.article_id in pop_dic:
            val = pop_dic[row.article_id]
        new_rows.append({"article_id":row.article_id, "popularity_score": val})

    df = pd.DataFrame(new_rows)

    sorted_score = df.sort_values(by='popularity_score', ascending=False)

    sorted_score['zpop'] = log4(np.arange(sorted_score.shape[0]) / 2.0 + 1.0)
    sorted_score = sorted_score.drop("popularity_score", axis=1)
    sorted_score.to_csv(experiment_directory + "/zpop_score.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--popularity_score', required=True)
    args = parser.parse_args()

    experiment_directory = args.experiment
    popularity_score = args.popularity_score
    new_xy_embedding = experiment_directory + "/new_xy_embeddings.csv"


    main(experiment_directory, pd.read_csv(popularity_score), pd.read_csv(new_xy_embedding))
