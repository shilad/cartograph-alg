import pandas as pd
import random

"""
This script creates the user study candidate labels for each map.
"""


def union_label_candidates(path, k, project):
    # get top k labels from both label algorithms and take the union
    all_labels = pd.read_csv(path + "/check.csv")
    all_labels['complex'] = all_labels['tf'] * all_labels['idf'] * all_labels['sum']
    all_labels['simple'] = all_labels['tf'] * all_labels['idf']
    all_labels = all_labels[['country', 'new_name', 'simple', 'complex']].drop_duplicates()
    simple = all_labels.sort_values('simple', ascending=False).groupby("country").head(k).sort_values('country')
    complex = all_labels.sort_values('complex', ascending=False).groupby("country").head(k).sort_values('country')
    union = pd.concat([simple, complex]).drop_duplicates().sort_values('country').reset_index().drop(columns=['index'])
    # add dummy labels
    dum = pd.read_csv("./data/" + project + "/dummy_labels.csv")
    used = set()
    dummy = pd.DataFrame()
    for k in range(7):
        i = random.randint(0, dum.shape[0] - 1)
        dum_label = dum.iloc[:, 1][i]
        for l in range(2):
            while dum_label in used:
                i = random.randint(0, dum.shape[0] - 1)
                dum_label = dum.iloc[:, 1][i]
            used.add(dum_label)
            dummy = dummy.append({'country': int(k), 'new_name': dum_label, 'simple': 0, 'complex': 0, 'isDummy': True}, ignore_index=True)

    u = pd.DataFrame(pd.concat([union, dummy]))[['country', 'new_name', 'simple', 'complex', 'isDummy']]
    u = u.astype({"country": int})
    u.to_csv(path + "/candidate_labels.csv")
    return union

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='path to the check.csv file')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--num_top_labels', required=True, type=int)
    parser.add_argument('--project', required=True)


    args = parser.parse_args()

    union_label_candidates(args.experiment, args.num_top_labels, args.project)


