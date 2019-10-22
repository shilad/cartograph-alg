import pandas as pd

from prior_work.regression.LabelModel import LabelModel

def main(experiment_dir):
    label_lst = ["h_cat", "key_phrases", "key_words", "lda", "links"]
    final = []

    for label_type in label_lst:
        file = experiment_dir + "/" + label_type + "_top_labels.csv"
        df = pd.read_csv(file)
        final.append(df)

    final = pd.concat(final)

    for index, row in final.iterrows():
        final.loc[index, "h_cat_tfidf"] = row['h_cat'] * row['tfidf']
        final.loc[index, "h_cat_pmi"] = row['h_cat'] * row['pmi']
        final.loc[index, "links_tfidf"] = row['links'] * row['tfidf']
        final.loc[index, "links_pmi"] = row['links'] * row['pmi']
        final.loc[index, "key_words_tfidf"] = row['key_words'] * row['tfidf']
        final.loc[index, "key_words_pmi"] = row['key_words'] * row['pmi']
        final.loc[index, "key_phrases_tfidf"] = row['key_phrases'] * row['tfidf']
        final.loc[index, "key_phrases_pmi"] = row['key_phrases'] * row['pmi']
        final.loc[index, "lda_tfidf"] = row['lda'] * row['tfidf']
        final.loc[index, "lda_pmi"] = row['lda'] * row['pmi']

    final = final.groupby('label', as_index=False).sum()
    final = final.fillna(0)
    rows = final.to_dict('records')
    model = LabelModel("/Users/senresearchlab/PycharmProjects/cartograph-alg/regression/bug_fixed.csv")
    for row in rows:
        row['avg_borda'] = model.predict(row)

    pd.DataFrame(rows).to_csv(experiment_dir + "/" + "predicted_borda.csv")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.exit()

    experiment_dir = sys.argv[1]
    main(experiment_dir)
