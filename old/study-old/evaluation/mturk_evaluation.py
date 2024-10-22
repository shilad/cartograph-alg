import pandas as pd
import re
from collections import defaultdict
import argparse
import json


PROJECTS = ['food', 'internet', 'media', 'technology']
LABEL_TYPES = ['h_cat', 'key_phrases', 'key_words', 'lda', 'links']
CLUSTER_TYPES = ['kmeans_plain', 'kmeans_augmented', 'LDA']
BLACK_LIST = ['food', 'foods', 'food and drink', 'media', 'medias', 'technology', 'no', 'none', 'nope']
USER_LABEL_BLACK_LIST = ['i can', 'i can\'t', 'i cannot', 'i could', 'i think', 'no ', 'none ', 'nope ', 'not really',
                         'notable figures', 'random things', 'strange and unfamiliar', 'i can think of',
                         'chosen are suitable', 'some better name', 'a good name', 'yes']


def get_most_popular_label_type(responses):
    """
    Input: Cleaned mturk responses
    Output: A default dictionary where keys = label type and values = number of points that label received
    Points: +5 for being selected as the #1 label for the group, +4 for being selected as #2, etc.
    """
    label_scores = defaultdict(int)
    for index, row in responses.iterrows():
        for column in responses.columns:
            if 'Answer.chosen_label' in column:
                project_num = re.findall("(?<=label_)(.*)(?=_)", column)[0]
                project = row['Input.project_' + project_num]
                if 'LDA' in row['Input.group_id_' + project_num]:
                    cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                else:
                    cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                if not row[column] == -1:
                    label = row['Input.label_' + str(project_num) + '_' + str(row[column])]
                    for type in LABEL_TYPES:
                        if type == 'lda' and cluster_type == 'LDA':
                            df = pd.read_csv('study/' + project + '/' + cluster_type +
                                             '/labels/LDA_labels/LDA_labels.csv')
                        else:
                            df = pd.read_csv('study/' + project + '/' + cluster_type + '/labels/' + type +
                                             '/top_labels.csv')
                        for col in df.columns[1:]:
                            df[str(col)] = df[str(col)].astype(str).str.lower()
                            if label in df[str(col)].values:
                                rank = re.findall('.$', column)[0]
                                label_scores[type] += 5 - int(rank) + 1
    return label_scores


def get_cluster_labels(responses):
    """
    Input: Cleaned mturk responses
    Output: A dictionary in the form: {project: {cluster_type: {cluster_number: {default dict}}}} where the keys in the
    default dict are labels for that cluster, and the values are the number of points that label received
    Points: +5 for being selected as the #1 label for the group, +4 for being selected as #2, etc.
    """
    cluster_labels = {}
    for proj in PROJECTS:
        cluster_labels[proj] = {}
        for cluster in CLUSTER_TYPES:
            cluster_labels[proj][cluster] = {}
            for i in range(7):
                cluster_labels[proj][cluster][i] = defaultdict(int)
    for index, row in responses.iterrows():
        for column in responses.columns:
            if 'Answer.chosen_label' in column:
                project_num = re.findall("(?<=label_)(.*)(?=_)", column)[0]
                project = row['Input.project_' + project_num]
                if 'LDA' in row['Input.group_id_' + project_num]:
                    cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                else:
                    cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                if not row[column] == -1:
                    label = row['Input.label_' + str(project_num) + '_' + str(row[column])]
                    cluster_num = re.findall("(?<=_c)(.*)(?=_)", row['Input.group_id_' + project_num])[0]
                    rank = re.findall('.$', column)[0]
                    cluster_labels[project][cluster_type][int(cluster_num)][label] += 5 - int(rank) + 1

    return cluster_labels


def get_top_cluster_labels(cluster_labels):
    """
    Input: Dictionary from get_cluster_labels()
    Output: A dictionary in the same form as cluster_labels, but the values for each cluster is a list of the top
    10 labels for that cluster (or less if that cluster did not have at least 10 labels that were selected as top
    choices by a user)
    """
    top_cluster_labels = {}
    for proj in cluster_labels:
        top_cluster_labels[proj] = {}
        for cluster in CLUSTER_TYPES:
            top_cluster_labels[proj][cluster] = {}
            for i in range(7):
                top_cluster_labels[proj][cluster][i] = []
    for proj in PROJECTS:
        for cluster_type in CLUSTER_TYPES:
            for i in range(7):
                sorted_labels = sorted(cluster_labels[proj][cluster_type][i].items(), key=lambda kv: kv[1], reverse=True)
                for label in sorted_labels:
                    if len(top_cluster_labels[proj][cluster_type][i]) >= 10:
                        break
                    else:
                        if label[0] not in BLACK_LIST:
                            top_cluster_labels[proj][cluster_type][i].append(label[0])
    return top_cluster_labels


def get_most_popular_cluster(responses):
    """
    Input: Cleaned mturk responses
    Output: Default dictionary where keys = type of cluster, values = number of points it received
    Points: +5 if user gave the cluster coherency a 5, +4 for a 4, etc.
    """
    cluster_scores = defaultdict(int)
    for index, row in responses.iterrows():
        for column in responses.columns:
            if 'Answer.cluster_coherency' in column:
                project_num = re.findall('.$', column)[0]
                if 'LDA' in row['Input.group_id_' + project_num]:
                    cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                else:
                    cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                score = row[column]
                cluster_scores[cluster_type] += score
    return cluster_scores


def clean_user_labels(responses):
    """
    Input: Cleaned mturk responses
    Output: Dictionary in the form {project: {cluster_type: {cluster_number: {set}}}} where the values in the set
    are the additional user-generated labels from the study (cleaned for useless answers)
    """
    user_labels, cleaned_labels = {}, {}
    for proj in PROJECTS:
        user_labels[proj], cleaned_labels[proj] = {}, {}
        for cluster in CLUSTER_TYPES:
            user_labels[proj][cluster], cleaned_labels[proj][cluster] = {}, {}
            for i in range(7):
                user_labels[proj][cluster][i], cleaned_labels[proj][cluster][i] = [], set()
    for index, row in responses.iterrows():
        for column in responses.columns:
            if 'Answer.other_label' in column:
                project_num = re.findall(".$", column)[0]
                project = row['Input.project_' + project_num]
                if 'LDA' in row['Input.group_id_' + project_num]:
                    cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                else:
                    cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                cluster_num = re.findall("(?<=_c)(.*)(?=_)", row['Input.group_id_' + project_num])[0]
                user_labels[project][cluster_type][int(cluster_num)].append(row[column])
    for project in PROJECTS:
        for cluster_type in CLUSTER_TYPES:
            for i in range(7):
                    for label in user_labels[project][cluster_type][i]:
                        if not type(label) == float:
                            label = label.lower().replace('.', '')
                            if label not in BLACK_LIST:
                                not_in = True
                                for phrase in USER_LABEL_BLACK_LIST:
                                    if phrase in label:
                                        not_in = False
                                if not_in:
                                    cleaned_labels[project][cluster_type][i].add(label)
    return cleaned_labels


def get_cluster_counts(responses):
    """
    Input:
    Output:
    """
    cluster_counts = {}
    for proj in PROJECTS:
        cluster_counts[proj] = {}
        for cluster in CLUSTER_TYPES:
            cluster_counts[proj][cluster] = {}
            for i in range(7):
                cluster_counts[proj][cluster][i] = defaultdict(int)
    for index, row in responses.iterrows():
        for column in responses.columns:
            if 'Answer.dont-belong' in column:
                project_num = re.findall("(?<=_)(.*)(?=_)", column)[0]
                project = row['Input.project_' + project_num]
                if 'LDA' in row['Input.group_id_' + project_num]:
                    cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                else:
                    cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                cluster_num = re.findall("(?<=_c)(.*)(?=_)", row['Input.group_id_' + project_num])[0]
                article_num = re.findall("(?<=\.)[^.]*$", column)[0]
                article_name = row['Input.article_' + project_num + '_' + article_num]
                if not row[column]:
                    cluster_counts[project][cluster_type][int(cluster_num)][article_name] += 1
    return cluster_counts


def get_gold_standard_clusters(cluster_counts):
    gold_standard_clusters = {}
    for proj in PROJECTS:
        gold_standard_clusters[proj] = {}
        for cluster in CLUSTER_TYPES:
            gold_standard_clusters[proj][cluster] = {}
            for i in range(7):
                gold_standard_clusters[proj][cluster][i] = []
    for project in PROJECTS:
        for cluster_type in CLUSTER_TYPES:
            for i in range(7):
                for article in cluster_counts[project][cluster_type][i]:
                    if cluster_counts[project][cluster_type][i][article] > 2:
                        gold_standard_clusters[project][cluster_type][i].append(article)
    return gold_standard_clusters


def main(responses):
    label_scores = get_most_popular_label_type(responses)
    print(str(json.dumps({"label_scores": label_scores})))

    cluster_labels = get_cluster_labels(responses)
    top_cluster_labels = get_top_cluster_labels(cluster_labels)

    for project in top_cluster_labels:
        for cluster_type in top_cluster_labels[project]:
            df = pd.DataFrame.from_dict(top_cluster_labels[project][cluster_type], orient='index')
            df['country'] = df.index
            df = df.set_index('country')
            df.to_csv('study/' + project + '/' + cluster_type + '/gold_standard_labels.csv')

    cluster_scores = get_most_popular_cluster(responses)
    print(str(json.dumps({"cluster_scores": cluster_scores})))

    cleaned_user_labels = clean_user_labels(responses)

    for project in cleaned_user_labels:
        for cluster_type in cleaned_user_labels[project]:
            df = pd.DataFrame.from_dict(cleaned_user_labels[project][cluster_type], orient='index')
            df['country'] = df.index
            df = df.set_index('country')
            df.to_csv('study/' + project + '/' + cluster_type + '/user_chosen_labels.csv')
    cluster_counts = get_cluster_counts(responses)
    gold_standard_clusters = get_gold_standard_clusters(cluster_counts)

    for project in gold_standard_clusters:
        for cluster_type in gold_standard_clusters[project]:
            df = pd.DataFrame.from_dict(gold_standard_clusters[project][cluster_type], orient='index')
            df['country'] = df.index
            df = df.set_index('country')
            df.to_csv('study/' + project + '/' + cluster_type + '/gold_standard_clusters.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation mturk study responses')
    parser.add_argument('--responses', required=True)

    args = parser.parse_args()

    mturk_responses = pd.read_csv(args.responses)
    main(mturk_responses)

# mturk_responses = pd.read_csv('study/evaluation/cleaned_mturk_results.csv')
# main(mturk_responses)