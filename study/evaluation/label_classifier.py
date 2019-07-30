import pandas as pd
import re
from collections import defaultdict


PROJECTS = ['food', 'internet', 'media', 'technology']
LABEL_TYPES = ['h_cat', 'key_phrases', 'key_words', 'lda', 'links']
CLUSTER_TYPES = ['kmeans_plain', 'kmeans_augmented', 'LDA']


def get_cluster_labels(responses):
    label_counts = {}
    for proj in PROJECTS:
        label_counts[proj] = {}
        for cluster in CLUSTER_TYPES:
            label_counts[proj][cluster] = {}
            for i in range(7):
                label_counts[proj][cluster][i] = defaultdict(int)
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
                    label_counts[project][cluster_type][int(cluster_num)][label] += 1
    return label_counts


def create_dataframe(responses, label_counts):
    columns = ['cluster_id', 'label', 'tfidf', 'pmi', 'h_cat', 'key_phrases', 'key_words', 'lda', 'links', 'score']
    df = pd.DataFrame(columns=columns)
    labels = set()
    x = 0
    for index, row in responses.iterrows():
        if x % 100 == 0:
            print(str(x) + ' rows completed')
        for column in responses.columns:
            label_row = []
            if 'Input.label' in column:
                if not str(row[column]) == 'nan':
                    if row[column] not in labels:
                        project_num = re.findall("(?<=label_)(.*)(?=_)", column)[0]
                        project = row['Input.project_' + project_num]
                        if 'LDA' in row['Input.group_id_' + project_num]:
                            cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                        else:
                            cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                        cluster_num = re.findall("(?<=_c)(.*)(?=_)", row['Input.group_id_' + project_num])[0]
                        cluster_id = project + '_' + cluster_type + '_' + cluster_num
                        label_row.extend([cluster_id, row[column], 1, 0])
                        label_type = get_label_type(row[column], project, cluster_type)
                        for i in range(5):
                            if label_type == LABEL_TYPES[i]:
                                label_row.extend([1])
                            else:
                                label_row.extend([0])
                        counts = label_counts[project][cluster_type][int(cluster_num)][row[column]]
                        label_row.extend([counts])
                        labels.add(row[column])

                        df_row = {}
                        for i in range(10):
                            df_row[columns[i]] = label_row[i]
                        df = df.append(df_row, ignore_index=True)

        x += 1
    return df


def get_label_type(label, project, cluster_type):
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
                return type
    return -1


def main(responses):
    label_counts = get_cluster_labels(responses)
    df = create_dataframe(responses, label_counts)
    df.to_csv('study/evaluation/label_matrix.csv')


mturk_responses = pd.read_csv('study/evaluation/cleaned_mturk_results.csv')
main(mturk_responses)
