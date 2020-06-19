import pandas as pd

"""
This script creates the user study candidate labels for each map. 
"""


def union_label_candidates(path, k):
    all_labels = pd.read_csv(path + "/check.csv")
    all_labels['complex'] = all_labels['tf'] * all_labels['idf'] * all_labels['sum']
    all_labels['simple'] = all_labels['tf'] * all_labels['idf']
    all_labels = all_labels[['country', 'new_name', 'simple', 'complex']].drop_duplicates()
    simple = all_labels.sort_values('simple', ascending=False).groupby("country").head(k).sort_values('country')
    complex = all_labels.sort_values('complex', ascending=False).groupby("country").head(k).sort_values('country')
    union = pd.concat([simple, complex]).drop_duplicates().sort_values('country').reset_index().drop(columns=['index'])
    union.to_csv(path + "/candidate_labels.csv")
    return union

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='path to the check.csv file')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--num_top_labels', required=True, type=int)

    args = parser.parse_args()

    union_label_candidates(args.experiment, args.num_top_labels)
