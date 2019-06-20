"""
Given a set of article names and labels from the cartograph algorithm, evaluates the quality of those labels
against the ground truth labels (defined by the WikiProjects Directory on Wikipedia).

Author: Lily Irvin
"""


from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import json


def find_matches(gold_standard, alg):
    exact_count = 0
    synonym_count = 0

    for index, row in gold_standard.iterrows():
        truth = row['label']
        truth_stems = []
        truth_synonyms = []

        if " " in truth:
            words = word_tokenize(truth)
            for word in words:

                stem = get_stem(word)
                if stem is not None:
                    truth_stems.append(stem)

                synonyms = get_synonyms(word)
                if synonyms is not None:
                    for synonym in synonyms:
                        truth_synonyms.append(synonym)

        alg_label = alg[str(index)]["Country"]

        if " " in alg_label:
            if (set(alg_label.split())).issubset(set(truth_stems)):
                exact_count += 1
                print(truth, alg_label, "exact")
            if (set(alg_label.split())).issubset(set(truth_synonyms)):
                synonym_count += 1
                print(truth, alg_label, "synonyms")

        else:
            if (set(alg_label)).issubset(set(truth_stems)):
                exact_count += 1
                print(truth, alg_label, "exact")
            if (set(alg_label)).issubset(set(truth_synonyms)):
                synonym_count += 1
                print(truth, alg_label, "synonym")
    return exact_count, synonym_count


def get_stem(word):
    ps = PorterStemmer()
    return ps.stem(word)


def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return synonyms


def main(gold_standard_csv, alg_json):
    gold_standard_labels = pd.read_csv(gold_standard_csv)
    with open(alg_json) as file:
        alg_labels = json.load(file)
    exact_matches, synonym_matches = find_matches(gold_standard_labels, alg_labels)
    print("There were " + str(exact_matches) + " articles labeled exactly.")
    print("There were " + str(synonym_matches) + " articles labeled with a synonym or synonyms of the true label.")


if __name__ == '__main__':
    main('../data/gold_standard_labels.csv', '../experiments/culture/0000/domain.json')
