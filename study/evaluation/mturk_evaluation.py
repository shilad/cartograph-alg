import pandas as pd
import re
from collections import defaultdict


PROJECTS = ['food', 'internet', 'media', 'technology']
LABEL_TYPES = ['h_cat', 'key_phrases', 'key_words', 'lda', 'links']
CLUSTER_TYPES = ['kmeans_plain', 'kmeans_augmented', 'LDA']
BLACK_LIST = ['food', 'foods', 'food and drink', 'media', 'medias', 'technology', 'no', 'none', 'nope']
USER_LABEL_BLACK_LIST = ['i can', 'i can\'t', 'i cannot', 'i could', 'i think', 'no ', 'none ', 'nope ', 'not really',
                         'notable figures', 'random things', 'strange and unfamiliar', 'i can think of',
                         'chosen are suitable', 'some better name', 'a good name', 'yes']


def get_most_popular_label(responses):
    """
    Input: Cleaned mturk responses
    Output: A default dictionary where keys = label type and values = number of points that label received
    Points: +5 for being selected as the #1 label for the group, +4 for being selected as #2, etc.
    """
    label_scores = defaultdict(int)
    x = 0
    for index, row in responses.iterrows():
        if x % 100 == 0:
            print(str(x) + ' rows completed')
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
        x += 1
    return label_scores


def get_cluster_labels(responses):
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
                x = 0
                while len(top_cluster_labels[proj][cluster_type][i]) < 5:
                    if sorted_labels[x][0] not in BLACK_LIST:
                        top_cluster_labels[proj][cluster_type][i].append(sorted_labels[x][0])
                    x += 1
    return top_cluster_labels


def get_most_popular_cluster(responses):
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


def main(responses):
    # label_scores = get_most_popular_label(responses)
    # print(label_scores)

    cluster_labels = get_cluster_labels(responses)
    top_cluster_labels = get_top_cluster_labels(cluster_labels)

    for project in top_cluster_labels:
        for cluster_type in top_cluster_labels[project]:
            df = pd.DataFrame.from_dict(top_cluster_labels[project][cluster_type], orient='index')
            df['country'] = df.index
            df = df.set_index('country')
            df.to_csv('study/' + project + '/' + cluster_type + '/gold_standard_labels.csv')

    cluster_scores = get_most_popular_cluster(responses)
    print(cluster_scores)

    cleaned_user_labels = clean_user_labels(responses)
    print(cleaned_user_labels)

    for project in cleaned_user_labels:
        for cluster_type in cleaned_user_labels[project]:
            df = pd.DataFrame.from_dict(cleaned_user_labels[project][cluster_type], orient='index')
            df.to_csv('study/' + project + '/' + cluster_type + '/user_chosen_labels.csv')


mturk_responses = pd.read_csv('study/evaluation/cleaned_mturk_results.csv')
main(mturk_responses)
