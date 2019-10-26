import pandas as pd
from beta.regression.LabelModel import LabelModel

def main(experiment_dir):
    label_lst = ["h_cat", "key_phrases", "key_words", "lda", "links"]
    final = []

    for label_type in label_lst:
        file = experiment_dir + "/" + label_type + "_top_labels.csv"
        df = pd.read_csv(file)
        final.append(df)

    final = pd.concat(final)
    final = final.fillna(0)
    final['h_cat_tfidf'] = final['h_cat'] * final['tfidf']
    final['key_words_tfidf'] = final['key_words'] * final['tfidf']
    final['key_phrases_tfidf'] = final['key_phrases'] * final['tfidf']
    final['links_tfidf'] = final['links'] * final['tfidf']
    final['lda_tfidf'] = final['lda'] * final['tfidf']


    final = final.groupby(['country', 'label_name']).sum().reset_index()
    final['label_name'] = final['label_name'].str.replace('_', ' ')

    final = final.fillna(0)
    rows = final.to_dict('records')
    model = LabelModel("./beta/regression/final_labels.csv")
    model.run_linear_regression()

    for row in rows:
        row['avg_borda'] = model.predict(row)

    pd.DataFrame(rows).to_csv(experiment_dir + "/" + "predicted_borda.csv")


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.exit()

    experiment_dir = sys.argv[1]
    main(experiment_dir)
