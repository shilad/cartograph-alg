import pandas as pd

def main(projects, algs):
    candidates = []
    for project in projects:
        for alg in algs:
            candidates.append(pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/" + project +"/"+ alg + '/candidate_labels.csv'))
    for alg in algs[0: len(algs)-1]:
        candidates.append(pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/media/" + alg + '/candidate_labels.csv'))

    final = pd.concat(candidates, ignore_index=True)
    final = final.iloc[:, 2:]
    print(final.columns)
    final.to_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/ml_final_labels.csv")
    borda = pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/hit_labels.csv")
    borda = borda[['name', 'share', 'avg_borda', 'cluster_alg']]
    # final= pd.read_csv("/Users/senresearch/PycharmProjects/cartograph-alg/study/ml_final_labels.csv")
    print(borda.columns)
    final= final.replace(' ', '_', regex=True)

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
#     main(list(projects), list(algs))