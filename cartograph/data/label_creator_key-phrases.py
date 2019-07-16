"""
    Given domain concepts, output keyword_names.csv (label id and label string) and
    article_keywords.csv (article id and label id)
    Author: Yuren "Rock" Pang and Lily Irvin
"""


from gensim.summarization import keywords, textcleaner
import pandas as pd
import os
import spacy
import gensim
import logging


nlp = spacy.load("en_core_web_sm")


def tokenize_sentences(domain_concept, summary):
    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ', '']
    for stop_word in my_stop_words:
        lexeme = nlp.vocab[stop_word]
        lexeme.is_stop = True

    sentences = []

    if not isinstance(summary, float):
        summary_sentences = textcleaner.split_sentences(summary)
        cleaned_sentences = []

        for sent in summary_sentences:
            cleaned_sentences.append(nlp(sent.lower()))

        for sentence in cleaned_sentences:
            sent = []
            for w in sentence:
                if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                    sent.append(w.lemma_.strip())
            sentences.append(sent)
    else:
        logging.warning(domain_concept + ": does not have summary")
        pass

    return sentences


def train_model(article_summaries):
    """
    Given the summary of each domain concept, remove stop words, punctuations, numbers, and newline,
    create a bag of words (called corpus) from the texts and then trains a gensim model with it.
    """
    corpus = []

    for row in article_summaries.itertuples():
        sentences = tokenize_sentences(row.article_name, row.extract)
        if sentences != []:
            corpus.extend(tokenize_sentences(row.article_name, row.extract))

    bigram = gensim.models.Phrases(corpus, min_count=30)
    trigram = gensim.models.Phrases(bigram[corpus], min_count=10)

    return bigram, trigram


def fetch_key_phrases(domain_concept, summary, bigram, trigram):
    key_phrases = set()
    sentences = tokenize_sentences(domain_concept, summary)

    for sentence in sentences:
        bigrams = bigram[sentence]
        trigrams = trigram[bigrams]
        key_phrases.update(bigrams + trigrams)

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    if not isinstance(summary, float):
        sum = nlp(summary)
        cleaned_summary = ''
        for w in sum:
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                    cleaned_summary += ' ' + w.text
        for word in keywords(cleaned_summary, lemmatize=True).split('\n'):
            key_phrases.add(word)

    return key_phrases


def create_labels(article_summaries, bigram, trigram):
    """
    Find the text of each domain concept and creates a data frame with articles and keyword labels
    :return: a dataframe with article id and label id
    """

    # mapping from ids to labels
    labels_to_id = {}
    rows_list = []
    x = 0

    for row in article_summaries.itertuples():
        if x % 1000 == 0:
            print(str(x) + ' articles completed')
        for keyword in fetch_key_phrases(row.article_name, row.extract, bigram, trigram):
            if keyword not in labels_to_id:
                labels_to_id[keyword] = len(labels_to_id)
            id = labels_to_id.get(keyword, len(labels_to_id))
            rows_list.append({"article_id": row.article_id, "label_id": id})
        x += 1
    return labels_to_id, pd.DataFrame(rows_list)


def create_label_id_str_csv(directory, labels_to_ids):
    id_to_label = [ (id, label) for (label, id) in labels_to_ids.items() ]
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/key-phrases_names.csv', index=False)


def create_article_label_csv(article_label_df, directory):
    article_label_df.to_csv(directory + "/article_key-phrases.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    article_summaries = pd.read_csv(map_directory + '/article_summary.csv')
    bigram, trigram = train_model(article_summaries)
    labels_to_id, label_df = create_labels(article_summaries, bigram, trigram)
    create_article_label_csv(label_df, map_directory)
    create_label_id_str_csv(map_directory, labels_to_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)