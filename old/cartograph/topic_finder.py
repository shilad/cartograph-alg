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

For topic model evaluation: https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
"""

import gensim
import spacy
import pandas as pd
import logging

from gensim.models import LdaModel, LsiModel, HdpModel
from gensim.corpora import Dictionary
from collections import OrderedDict
import os


nlp = spacy.load('en_core_web_sm')


def create_text_corpus_by_csv(article_text_csv):

    """
    Given the summary of each domain concept, remove stop words, punctuations, numbers, and newline,
    create a bag of words (called corpus) from the texts.
    To get bigram, gensim.models.Phrases() is performed. To get trigram, perform Phrases() on the bigram again
    bigram = gensim.models.Phrases(texts) # text is a list of list
    bigram[texts[0]]  # display the word (bigram) in the first article text
    Reference: https://radimrehurek.com/gensim/models/phrases.html
    :param article_summary_csv:
    :return:
    """
    df = pd.read_csv(article_text_csv, encoding='UTF-8')

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    texts, article = [], []
    id_to_article_id = {}

    for row in df.itertuples():
        if not isinstance(row.text, float):
            id_to_article_id.update({len(id_to_article_id):int(row.article_id)})  # assign id
            summary = nlp(row.text.lower())
            for w in summary:
                if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                    article.append(w.lemma_.strip())
            texts.append(article)
            article = []
        else:
            logging.warning("(%d, %s): does not have summary", row.article_id, row.article_name)
            pass

    bigram = gensim.models.Phrases(texts, threshold=8)
    texts = [bigram[line] for line in texts]

    trigram = gensim.models.Phrases(texts)
    texts = [trigram[line] for line in texts]

    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.7)
    corpus = [dictionary.doc2bow(text) for text in texts]

    return corpus, dictionary, id_to_article_id


def run_model(corpus, dictionary, method='LDA', num_topics=8):
    if method == 'LDA':
        ldamodel = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, iterations=400, passes=20)
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


def create_topic_df(model, num_words):
    """
    The output format is: topic, word1, prob1, word2, prob2 ...
    :param model:
    :return:
    """
    # the topic format [(0, '0.337*"food" + ...'), (1, '0.123*"product"...')
    topics = model.show_topics(num_words=int(num_words))

    row_dic = OrderedDict()
    row_list = []

    for topic in topics:
        row_dic.update({'country': topic[0]})
        topic_with_prob_str = topic[1].split(' + ')
        for index, label in enumerate(topic_with_prob_str):
            prob_label = label.split('*')
            row_dic.update({'label' + str(index): prob_label[1].strip("\"")})
            row_dic.update({'prob' + str(index): float(prob_label[0].strip())})
        row_list.append(row_dic)
        row_dic = OrderedDict()
    return pd.DataFrame(row_list)


def create_article_df(model, corpus, id_to_article_id):
    """
    The output format is similar the article_labels where the same article_id with another topic follows the previous one
    :param model:
    :param corpus:
    :param id_to_article_id:
    :return:
    """
    row_list = []

    for index, article in enumerate(corpus):
        document_topics = model.get_document_topics(article)
        for prob_tuple in document_topics:
            row_list.append({'article_id': id_to_article_id[index],
                             'country': prob_tuple[0],
                             'prob': prob_tuple[1]})
    return pd.DataFrame(row_list)


def main(directory, data, num_words, label_file, article_file):
    if not os.path.exists(directory):
        os.makedirs(directory)

    label_directory = directory + '/labels/LDA_labels'
    if not os.path.exists(label_directory):
        os.makedirs(label_directory)

    corpus, dictionary, id_to_article_id = create_text_corpus_by_csv(data + "/article_text.csv")
    num_topics = 7
    model = run_model(corpus, dictionary, method='LDA', num_topics=num_topics)
    topic_label_distribution_df = create_topic_df(model, num_words)
    article_topic_distribution_df = create_article_df(model, corpus, id_to_article_id)

    topic_label_distribution_df.to_csv(label_directory + label_file, index=False)
    article_topic_distribution_df.to_csv(directory + article_file, index=False)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in model.top_topics(corpus)]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    # Save the model
    model.save(directory + '/topic_model')


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 6:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory, data, num_words, label_file, article_file = sys.argv[1:]
    main(directory, data, num_words, label_file, article_file)
