import pandas as pd

def main(projects, algs):
    candidates = []
    for project in projects:
        for alg in algs:
            candidates.append(pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/" + project +"/"+ alg + '/candidate_labels.csv'))

    # for alg in algs[0: len(algs)-1]:
    #     candidates.append(pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/media/" + alg + '/candidate_labels.csv'))
    print(candidates[0]['lda_label'])
    final = pd.concat(candidates)
    final = final.iloc[:, 2:]
    print(final.columns)
    # final.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/ml_final_labels.csv")

    borda = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/hit_labels.csv")
    borda = borda[['name', 'share', 'avg_borda', 'cluster_alg']]
    print(borda.columns)
    final= final.replace(' ', '_', regex=True)
    print(final[final['lda_label']>0])
    for index, row in final.iterrows():
        final.loc[index, "h_cat_tfidf"] = row['h_cat'] * row['tfidf']
        final.loc[index, "h_cat_pmi"] = row['h_cat'] * row['pmi']
        final.loc[index, "links_tfidf"] = row['links'] * row['tfidf']
        final.loc[index, "links_pmi"] = row['links'] * row['pmi']
        final.loc[index, "key_words_tfidf"] = row['key_words'] * row['tfidf']
        final.loc[index, "key_words_pmi"] = row['key_words'] * row['pmi']
        final.loc[index, "key_phrases_tfidf"] = row['key_phrases'] * row['tfidf']
        final.loc[index, "key_phrases_pmi"] = row['key_phrases'] * row['pmi']
        final.loc[index, "lda_tfidf"] = row['lda_label'] * row['tfidf']
        final.loc[index, "lda_pmi"] = row['lda_label'] * row['pmi']
    final.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/ml_final_labels.csv")
    merged = final.drop(columns=['h_cat', 'links', 'lda_label', 'key_words', 'key_phrases'])

    merged = merged.groupby('label', as_index=False).sum()

    print(merged['label'])


    merged = pd.merge(final, borda, how="inner", right_on=("name", "cluster_alg"), left_on=("label", "cluster_alg"))
    merged = merged.drop(columns=['cluster_alg', 'name' ])
    merged = merged.drop_duplicates()
    print(merged['label'])

    merged.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/labels_for_ml_study.csv")

projects = ["food", "internet", "technology"]
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