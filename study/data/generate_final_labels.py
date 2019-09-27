import pandas as pd

def main(directory, alg):
    categories = pd.read_csv(directory + '/labels/h_cat/top_labels.csv')
    key_phrases = pd.read_csv(directory + '/labels/key_phrases/top_labels.csv')
    key_words = pd.read_csv(directory + '/labels/key_words/top_labels.csv')
    links = pd.read_csv(directory + '/labels/links/top_labels.csv')
    if alg != "LDA":
        lda = pd.read_csv(directory + '/labels/lda/top_labels.csv')
    else:
        lda = pd.read_csv(directory + '/labels/LDA_labels/top_labels.csv')

    final = pd.concat([categories, key_phrases, key_words, links, lda], ignore_index=True)
    print(final)
    final.to_csv(directory + '/candidate_labels.csv')


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)
    directory, alg = sys.argv[1:]
    main(directory, alg)