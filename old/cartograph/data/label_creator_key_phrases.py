"""
    Given domain concepts, output keyword_names.csv (label id and label string) and
    article_keywords.csv (article id and label id)
    Author: Yuren "Rock" Pang and Lily Irvin
    To run spacy, do: python -m spacy download en_core_web_sm
"""
from collections import defaultdict
from gensim import models
from gensim.summarization import keywords, textcleaner
import pandas as pd
import os
import spacy
import gensim
import logging


MAX_PHRASES_PER_ARTICLE = 100  # max phrases retained per article

nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger'])


def tokenize_sentences(domain_concept, text):
    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ', '']
    for stop_word in my_stop_words:
        lexeme = nlp.vocab[stop_word]
        lexeme.is_stop = True

    sentences = []

    if not isinstance(text, float):
        text_sentences = textcleaner.split_sentences(text)
        cleaned_sentences = []

        for sent in text_sentences:
            cleaned_sentences.append(nlp(sent.lower()))

        for sentence in cleaned_sentences:
            sent = []
            for w in sentence:
                if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                    sent.append(w.text.strip())
            sentences.append(sent)
    else:
        logging.warning(domain_concept + ": does not have summary")
        pass

    return sentences


def phrasify(article_text):
    """
    Given the summary of each domain concept, remove stop words, punctuations, numbers, and newline,
    Use gensim to identify phrases within the text. Returns a list of documents, where each document
    is a list of phrases in the document.
    """
    corpus = []

    for row in article_text.itertuples():
        sentences = tokenize_sentences(row.article_name, row.text)
        corpus.extend([(row.article_id, row.article_name, s) for s in sentences])
    sentences = [doc[-1] for doc in corpus]

    bigram = gensim.models.Phrases(sentences, min_count=30)
    phrases1 = list(bigram[sentences])
    trigram = gensim.models.Phrases(phrases1, min_count=10)
    phrases2 = list(trigram[phrases1])

    assert len(phrases2) == len(phrases1) == len(corpus)

    phrase_corpus = [(doc[0], doc[1], sentence) for (doc, sentence) in zip(corpus, phrases2)]

    return phrase_corpus


def fetch_key_phrases(domain_concept, text, bigram, trigram):
    key_phrases = set()
    sentences = tokenize_sentences(domain_concept, text)

    for sentence in sentences:
        bigrams = bigram[sentence]
        trigrams = trigram[bigrams]
        key_phrases.update(bigrams + trigrams)

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    if not isinstance(text, float):
        sum = nlp(text)
        cleaned_summary = ''
        for w in sum:
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                    cleaned_summary += ' ' + w.text
        for word in keywords(cleaned_summary, lemmatize=True).split('\n'):
            key_phrases.add(word)

    return key_phrases


def create_labels(phrase_corpus):
    """
    Find the text of each domain concept and creates a data frame with articles and keyword labels
    :return: a dataframe with article id and label id
    """

    article_id_to_name = {}
    article_word_freq = defaultdict(lambda: defaultdict(float))  # article id -> word id -> freq
    vocab = {}  # word -> word id
    for article_id, article_name, article_text in phrase_corpus:
        article_id_to_name[article_id] = article_name
        for word in article_text:
            if word not in vocab:
                vocab[word] = len(vocab)
            word_id = vocab[word]
            article_word_freq[article_id][word_id] += 1

    sparse_matrix = []
    for article_id in sorted(article_word_freq.keys()):
        sparse_row = []
        for word_id in sorted(article_word_freq[article_id].keys()):
            sparse_row.append((word_id, article_word_freq[article_id][word_id]))
        sparse_matrix.append((article_id, article_id_to_name[article_id], sparse_row))

    index2word = {i : w for (w, i) in vocab.items()}
    tfidf = models.TfidfModel([row for (id, name, row) in sparse_matrix])
    results = []
    for id, name, row in sparse_matrix:
        tfidf_row = tfidf[row]
        tfidf_row.sort(key=lambda x: x[1], reverse=True)
        results.append((id, name, tfidf_row[:MAX_PHRASES_PER_ARTICLE]))

    return index2word, results


def create_label_id_str_csv(directory, index2word):
    id_to_label = list(index2word.items())
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/keyphrases_names.csv', index=False)


def create_article_label_csv(tf_idf_results, directory):
    rows = []
    for (article_id, article_name, tfidf) in tf_idf_results:
        for (word_id, score) in tfidf:
            rows.append({'article_id' : article_id, 'label_id' : word_id, 'score' : score})
    df = pd.DataFrame(rows)
    df.to_csv(directory + "/article_keyphrases.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    article_text = pd.read_csv(map_directory + '/article_text.csv')
    phrase_corpus = phrasify(article_text)
    index2word, tf_idf_results = create_labels(phrase_corpus)
    create_article_label_csv(tf_idf_results, map_directory)
    create_label_id_str_csv(map_directory, index2word)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
