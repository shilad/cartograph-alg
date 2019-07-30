"""
A script used to evaluate labels against user study results. The lower the score, the better the label results.

Author: Lu Li
"""
import argparse
import pandas as pd
import numpy as np
from pandas._libs import json


def evaluate_labels(gold_standard, label_set, k):
    gold_standard_labels = pd.read_csv(gold_standard)
    label_set = list(pd.read_csv(label_set)['label_name'].values)
    rank_list = []
    match = 0
    score = 0
    for i in range(len(label_set)):
        # print(gold_standard_labels.values)
        if label_set[i].lower() in gold_standard_labels.values:
            rank = np.where(gold_standard_labels.values == label_set[i].lower())[1][0]
            rank_list.append(rank)
            match += 1
            score += rank / 5
        else:
            score += 1.2
    return match, rank_list, score/k


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--experiment_directory', required=True)
    parser.add_argument('--gold_standard', required=True)
    parser.add_argument('--label_set', required=True)
    parser.add_argument('--k', required=True, type=int)

    args = parser.parse_args()


num_exact_matchs, rank_list, label_score = evaluate_labels(args.gold_standard, args.label_set, args.k)

print(str(json.dumps(num_exact_matchs)))
# print(str(json.dumps(rank_list)))
print(str(label_score))

