import argparse
import pandas as pd

def main(experiment_dir, borda_file):
    df = pd.read_csv(experiment_dir + "/" + borda_file)
    df_top_borda = df.loc[df.reset_index().groupby(['country'])['new_tfidf'].idxmax()]
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)

    print(df_top_borda)

    lst = []
    for index, row in df_top_borda.iterrows():
        lst.append({"country" : row["country"], "label_name" : row["new_label"]})

    pd.DataFrame(lst).to_csv(experiment_dir + "/final_labels.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for label picker.')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--borda_file', required=True)
    args = parser.parse_args()

    main(args.experiment, args.borda_file)
