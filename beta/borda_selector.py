import argparse
import pandas as pd
import numpy as np

def main(experiment_dir, borda_file):
    df = pd.read_csv(experiment_dir + "/" + borda_file)
    df_top_borda = df.sort_values('avg_borda', ascending=False).groupby(["country"]).head(1)

    lst = []
    for index, row in df_top_borda.iterrows():
        lst.append({"country" : row["country"], "label_name" : row["label_name"]})

    pd.DataFrame(lst).to_csv(experiment_dir + "/final_labels.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for label picker.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--borda_file', required=True)
    args = parser.parse_args()

    main(args.experiment, args.borda_file)