import pandas as pd


def get_top_labels_kmeans(labels):
    top_labels = [set() for x in range(7)]
    for label_type in labels:
        for row in label_type.itertuples():
            for i in range(1, 6):
                top_labels[row.country].add(label_type[str(i)].iloc[row.country])

    return top_labels


def get_top_labels_LDA(labels, lda_labels):
    top_labels = [set() for x in range(7)]
    for label_type in labels:
        for row in label_type.itertuples():
            for i in range(1, 6):
                top_labels[row.country].add(label_type[str(i)].iloc[row.country])
    for row in lda_labels.itertuples():
        for i in range(5):
            top_labels[row.country].add(lda_labels['label' + str(i)].iloc[row.country])

    return top_labels


def main(directory, clustering):
    categories = pd.read_csv(directory + '/labels/h_cat/top_labels.csv')
    key_phrases = pd.read_csv(directory + '/labels/key_phrases/top_labels.csv')
    key_words = pd.read_csv(directory + '/labels/key_words/top_labels.csv')
    links = pd.read_csv(directory + '/labels/links/top_labels.csv')

    if clustering == 'LDA':
        LDA_LDA = pd.read_csv(directory + '/labels/LDA_labels/LDA_labels.csv')

        top_labels_LDA = get_top_labels_LDA([categories, key_phrases, key_words, links], LDA_LDA)
        top_labels_LDA_df = pd.DataFrame(top_labels_LDA)
        top_labels_LDA_df.to_csv(directory + '/label_candidates.csv')

    else:
        LDA_kmeans = pd.read_csv(directory + '/labels/lda/top_labels.csv')

        top_labels_kmeans = get_top_labels_kmeans([categories, key_phrases, key_words, LDA_kmeans, links])
        top_labels_kmeans_df = pd.DataFrame(top_labels_kmeans)
        top_labels_kmeans_df.to_csv(directory + '/label_candidates.csv')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory, clustering = sys.argv[1:]
    main(directory, clustering)
