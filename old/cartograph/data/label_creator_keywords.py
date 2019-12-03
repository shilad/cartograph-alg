"""
    Given the text of a given set of domain concepts, output keyword_names.csv (label id and label string) and
    article_keywords.csv (article id and label id)
    Author: Yuren "Rock" Pang and Lily Irvin
"""

from gensim.summarization import keywords
import pandas as pd
import os
import spacy


nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger'])


def fetch_keywords(text):
    """
    Given the text of a domain concept, cleans the text (gets rid of stopwords/punctuation/numbers/etc.) and outputs
    the keywords from that text according to gensim's summarization algorithm.
    :param text of a given domain concept
    :return: List of keywords for that article
    """
    key_words = []

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stop_word in my_stop_words:
        lexeme = nlp.vocab[stop_word]
        lexeme.is_stop = True

    if not isinstance(text, float):
        cleaned_text = ''
        for w in nlp(text):
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                cleaned_text += ' ' + w.text

        for word, score in keywords(cleaned_text, scores=True):
            key_words.append((word, score))

    return key_words


def create_labels(article_text):
    """
    Find the text of each domain concept and creates a data frame with articles and keyword labels
    :return: a dataframe with article id and label id
    """

    # mapping from ids to labels
    labels_to_id = {}
    rows_list = []
    x = 0

    for row in article_text.itertuples():
        if x % 1000 == 0:
            print(str(x) + ' articles completed')
        for keyword, score in fetch_keywords(row.text):
            if keyword not in labels_to_id:
                labels_to_id[keyword] = len(labels_to_id)
            id = labels_to_id.get(keyword, len(labels_to_id))
            rows_list.append({"article_id": row.article_id, "label_id": id, "score": score})
        x += 1
    return labels_to_id, pd.DataFrame(rows_list)


def create_article_label_csv(article_label_df, directory):
    article_label_df.to_csv(directory + "/article_keywords.csv", index=False)


def create_label_id_str_csv(directory, labels_to_ids):
    id_to_label = [(id, label) for (label, id) in labels_to_ids.items()]
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/keyword_names.csv', index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    article_text = pd.read_csv(map_directory + '/article_text.csv')
    labels_to_id, label_df = create_labels(article_text)
    create_article_label_csv(label_df, map_directory)
    create_label_id_str_csv(map_directory, labels_to_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
