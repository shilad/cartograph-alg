"""
Given a article summary of domain concepts, save the topic model in the target directory,
print the representation for selected topics;

The scripts also include a way to incorporate labels from various sources (links, keyword, categories)
and perform topic modeling (create_article_label_candidates(), create_text_corpus_combined())

To use the saved model, simply use LdaModel.load(topic_model.model) and use functions such as
model.show_topics(), get_document_topics()

Author: Yuren 'Rock' Pang
Reference: Bhargav Srinivasa Desikan,
https://github.com/bhargavvader/personal/tree/master/notebooks/text_analysis_tutorial
"""

import gensim
import spacy
import pandas as pd
import logging
import argparse

from gensim.models import LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
from spacy.lang.en.stop_words import STOP_WORDS


def create_article_label_candidates(article_labels, label_names, cluster_groups, country_labels):
    my_stop_words = ['say', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', ' ', 'des', 'Years']
    STOP_WORDS.update(my_stop_words)

    article_labels_with_cluster = pd.merge(article_labels, cluster_groups, on='article_id').drop(['article_id', 'distance'], 1)

    text = [[] for i in range(len(country_labels))]
    label_names_dic = label_names.set_index('label_id')['label'].to_dict()

    for row in article_labels_with_cluster.itertuples():
        country = row.country
        label_word = label_names_dic[row.label_id]

        if label_word not in STOP_WORDS and not isinstance(label_word, float):

            text[country].append(label_word.lower().strip())

    print(text)
    return text


def create_text_corpus_combined(article_labels, label_names, cluster_groups, country_labels):
    texts = create_article_label_candidates(article_labels, label_names, cluster_groups, country_labels)

    bigram = gensim.models.Phrases(texts)
    texts = [bigram[line] for line in texts]

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus, dictionary


def create_text_corpus_by_csv(article_summary_csv):
    df = pd.read_csv(article_summary_csv, encoding='UTF-8')
    nlp = spacy.load('en')

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    texts, article = [], []

    for row in df.itertuples():
        if not isinstance(row.extract, float):
            summary = nlp(row.extract.lower())
            for w in summary:
                if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and w.text != 'I':
                    article.append(w.lemma_.strip())
            texts.append(article)
            article = []
        else:
            logging.warning("(%d, %s): does not have summary", row.article_id, row.article_name)
            pass

    bigram = gensim.models.Phrases(texts)
    texts = [bigram[line] for line in texts]

    trigram = gensim.models.Phrases(texts)
    texts = [trigram[line] for line in texts]

    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus, dictionary


def run_model(corpus, dictionary, method='LDA', num_topics=8):
    if method == 'LDA':
        ldamodel = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        return ldamodel
    elif method == 'LSI':
        lsimodel = LsiModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        return lsimodel
    elif method == 'HDP': # extension of LDA when # of topic is unknown
        hdpmodel = HdpModel(corpus=corpus, id2word=dictionary)
        return hdpmodel
    # Generate html vis
    # vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
    # pyLDAvis.save_html(vis, 'lda_vis_text.html')


def main(directory, experiment_path):
    corpus, dictionary = create_text_corpus_by_csv(directory + "/article_summary.csv")
    model = run_model(corpus, dictionary, method='LDA')
    print(model.show_topics(num_words=5))
    # Save the model
    model.save(experiment_path + 'topic_model')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select labels for cluster.')
    parser.add_argument('--data_folder', required=True)
    parser.add_argument('--experiment', required=True)

    args = parser.parse_args()

    directory = args.data_folder
    path = args.experiment

    main(directory, path)