import pandas as pd
import re
from collections import defaultdict
import argparse


PROJECTS = ['food', 'internet', 'media', 'technology']
LABEL_TYPES = ['h_cat', 'key_phrases', 'key_words', 'lda', 'links']
CLUSTER_TYPES = ['kmeans_plain', 'kmeans_augmented', 'LDA']


def get_labels(responses):
    label_votes = defaultdict(lambda: defaultdict(int))
    label_exposures = defaultdict(lambda: defaultdict(int))

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
                    cluster_id = re.findall("(?<=_c)(.*)(?=_)", row['Input.group_id_' + project_num])[0]
                    group_id = project + '_' + cluster_type + '_' + str(cluster_id)
                    label_votes[group_id][label] += 1
            if 'Input.label' in column:
                if not str(row[column]) == 'nan':
                    project_num = re.findall("(?<=label_)(.*)(?=_)", column)[0]
                    project = row['Input.project_' + project_num]
                    if 'LDA' in row['Input.group_id_' + project_num]:
                        cluster_type = re.findall("^[^_]+(?=_)", row['Input.group_id_' + project_num])[0]
                    else:
                        cluster_type = re.findall("^[^_]*_[^_]*", row['Input.group_id_' + project_num])[0]
                    cluster_id = re.findall("(?<=_c)(.*)(?=_)", row['Input.group_id_' + project_num])[0]
                    group_id = project + '_' + cluster_type + '_' + str(cluster_id)
                    label_exposures[group_id][row[column]] += 1

    rows = []
    for group_id in label_votes:
        for label in label_votes[group_id]:
            row = {
                'group_id': group_id,
                'label': label,
                'score': label_votes[group_id][label] / label_exposures[group_id][label]
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    return df


def get_label_scores():
    cluster_label_info = defaultdict(lambda: defaultdict(dict))
    for project in PROJECTS:
        for cluster_type in CLUSTER_TYPES:
            for type in LABEL_TYPES:
                if type == 'lda' and cluster_type == 'LDA':
                    df = pd.read_csv('study/' + project + '/LDA/labels/LDA_labels/LDA_labels.csv')
                    for cluster_id, row in df.iterrows():
                        group_id = project + '_' + cluster_type + '_' + str(cluster_id)
                        for i in range(10):
                            label = row['label' + str(i)].lower()
                            prob = row['prob' + str(i)]
                            cluster_label_info[group_id][label]['lda_prob'] = prob
                else:
                    df = pd.read_csv('study/' + project + '/' + cluster_type + '/labels/' + type +
                                     '/top_labels.csv')
                    for cluster_id, row in df.iterrows():
                        group_id = project + '_' + cluster_type + '_' + str(cluster_id)
                        for i in range(1, 11):
                            label = row[str(i)].lower()
                            tfidf = row['tfidf_' + str(i)]
                            pmi = row['pmi_' + str(i)]
                            cluster_label_info[group_id][label][type + '_tfidf'] = tfidf
                            cluster_label_info[group_id][label][type + '_pmi'] = pmi

    rows = []
    for group_id in cluster_label_info:
        for label in cluster_label_info[group_id]:
            scores = cluster_label_info[group_id][label]
            row = {
                'group_id' : group_id,
                'label' : label,
                'lda_prob' : scores.get('lda_prob', 0.0)
            }
            for type in LABEL_TYPES:
                for suffix in ('pmi', 'tfidf'):
                    key = type + '_' + suffix
                    row[key] = scores.get(key, 0.0)
            rows.append(row)
    df = pd.DataFrame(rows)
    return df


def main(responses):
    scores_df = get_label_scores()
    label_df = get_labels(responses)
    label_matrix = pd.merge(scores_df, label_df, on=['group_id', 'label'], how='outer')
    print(label_matrix['h_cat_tfidf'].isna().sum())
    print(label_matrix)
    label_matrix.to_csv('study/evaluation/label_matrix.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation mturk study responses')
    parser.add_argument('--responses', required=True)

    args = parser.parse_args()

    mturk_responses = pd.read_csv(args.responses)
    main(mturk_responses)


# mturk_responses = pd.read_csv('study/evaluation/cleaned_mturk_results.csv')
# main(mturk_responses)
