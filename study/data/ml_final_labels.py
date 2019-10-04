import pandas as pd

from regression.LabelModel import LabelModel


def main(projects, algs):
    candidates = []
    for project in projects:
        for alg in algs:
            candidates.append(pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/" + project +"/"+ alg + '/candidate_labels.csv'))

    # print(candidates[0]['lda_label'])
    final = pd.concat(candidates)
    final = final.iloc[:, 2:]
    print(final.columns)

    borda = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/hit_labels.csv")
    borda = borda[['name', 'share', 'avg_borda', 'cluster_alg', 'cluster_num']]
    print(borda.columns)
    final= final.replace(' ', '_', regex=True)
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
    final.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/ml_final_labels.csv")
    merged = final.drop(columns=['h_cat', 'links', 'lda', 'key_words', 'key_phrases'])

    merged = merged.groupby('label', as_index=False).sum()

    print(merged['label'])


    merged = pd.merge(final, borda, how="inner", right_on=("name", "cluster_alg", "cluster_num" ), left_on=("label", "alg", "country"))
    merged = merged.drop(columns=['cluster_alg', 'name' ])
    merged = merged.drop_duplicates()
    print(merged['label'])

    merged.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/labels_for_ml_study.csv")

    rows = merged.set_index('label')['value'].to_dict(orient='list')

    print(type(rows))
    model = LabelModel("/Users/senresearch/PycharmProjects/cartograph-alg/regression/bug_fixed.csv")
    for row in rows:
        row['avg_borda'] = model.predict(row)

    rows.to_csv("results.csv")

projects = ["food"]
algs = ["kmeans_augmented", "kmeans_plain", "LDA"]
main(projects, algs)

# if __name__ == '__main__':
#     import sys
#
#     if len(sys.argv) != 3:
#         sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
#         sys.exit(1)
#
#     projects, algs = sys.argv[1:]
#     main(list(projects), l
#     ist(algs))