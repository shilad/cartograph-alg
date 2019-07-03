"""
    Given domain concepts, output keyword_names.csv (label id and label string) and
    article_keywords.csv (article id and label id)
    Author: Yuren "Rock" Pang and Lily Irvin
"""

import wikipediaapi
from gensim.summarization import keywords
import pandas as pd
import os
import spacy
import gensim


nlp = spacy.load("en_core_web_sm")


def fetch_key_phrases(domain_concept):
    key_phrases = set()
    wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
    page = wiki_wiki.page(domain_concept).text.replace('\n', '')

    sentences = []
    doc = nlp(page)
    for sent in doc.sents:
        sentences.append(sent.text)

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    article, text = [], []
    for sentence in sentences:
        for w in nlp(sentence.lower()):
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                article.append(w.lemma_.strip())
        text.append(article)
        article = []

    bigram = gensim.models.Phrases(text, threshold=3)
    text = [bigram[line] for line in text]

    trigram = gensim.models.Phrases(text, threshold=2)
    text = [trigram[line] for line in text]

    for sentence in text:
        for word in sentence:
            if '_' in word:
                key_phrases.add(word)

    for word in keywords(page, lemmatize=True).split('\n'):
        key_phrases.add(word)

    return key_phrases


def create_labels(domain_concept_csv):
    """
    Find the text of each domain concept and creates a data frame with articles and keyword labels
    :return: a dataframe with article id and label id
    """

    # mapping from ids to labels
    labels_to_id = {}
    df = pd.read_csv(domain_concept_csv)
    rows_list = []
    x = 0
    # Loop through the domain concepts
    for index, row in df.iterrows():  # test df.head(10).iterrows()
        if x % 1000 == 0:
            print(str(x) + ' articles completed')
        article_id = row[0]
        domain_concept = row[1]
        for keyword in fetch_key_phrases(domain_concept):
            if keyword not in labels_to_id:
                labels_to_id[keyword] = len(labels_to_id)
            id = labels_to_id.get(keyword, len(labels_to_id))
            rows_list.append({"article_id": article_id, "label_id": id})
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

    labels_to_id, label_df = create_labels(map_directory + "/domain_concept.csv")
    create_article_label_csv(label_df, map_directory)
    create_label_id_str_csv(map_directory, labels_to_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)