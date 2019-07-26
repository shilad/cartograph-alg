import argparse

import pandas as pd
import numpy as np

def get_num_of_exact_matches(gold_standard, label_set):
    gold_standard_labels = pd.read_csv(gold_standard)
    print(gold_standard_labels)
    label_set = list(pd.read_csv(label_set)['country'].values)
    exact = 0
    for i in range(len(label_set)):
        if label_set[i] in gold_standard_labels:
            exact += 1
    return exact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='evaluate')
    parser.add_argument('--experiment_directory', required=True)
    parser.add_argument('--gold_standard', required=True)
    parser.add_argument('--label_set', required=True)
    args = parser.parse_args()


print(get_num_of_exact_matches(args.gold_standard, args.label_set))