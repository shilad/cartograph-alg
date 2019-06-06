import pandas as pd


def filter_articles(article_labels_csv, article_vectors_csv):
    """Filters out articles that do not have a navigation vector from the article_labels data frame."""
    article_labels = pd.read_csv(article_labels_csv)
    article_vectors = pd.read_csv(article_vectors_csv)
    for index, row in article_labels.iterrows():
        if row['article_id'] not in article_vectors['article_id']:
            article_labels.drop([row['article_id']])
    print(article_labels['article_id'].nunique())
    return article_labels


def create_csv(labels, directory):
    df = pd.DataFrame(labels, columns=['article_id', 'label_id'])
    df.to_csv(directory + '/filtered_article_labels.csv', index=False)


def main(map_directory):
    filtered = filter_articles(map_directory + '/article_labels.csv', map_directory + '/article_vectors.csv')
    create_csv(filtered, map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
