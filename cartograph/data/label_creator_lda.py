import pandas as pd


def create_topic_label_dic(topic_label_df):
    topic_label_dic = {}
    for index, row in topic_label_df.iterrows():
        for column in range(1, topic_label_df.shape[1]-1, 2):
            topic_id = topic_label_df.iloc[index, 0]
            label = topic_label_df.iloc[index, column]
            prob = topic_label_df.iloc[index, column+1]
            topic_label_dic.setdefault(topic_id, {}).update({label: prob})
    return topic_label_dic


def create_labels(article_id_word_dic):
    row_list = []
    labels_to_id = {}

    for article, word_prob_dic in article_id_word_dic.items():
        length_limit = 100 if len(word_prob_dic) > 100 else len(word_prob_dic)
        candidate_labels = sorted(word_prob_dic.items(), key=lambda x:x[1], reverse=True)  # find the first 100 candidates

        for index in range(length_limit):
            try:
                label = candidate_labels[index][0]
            except IndexError:
                print(candidate_labels)

            prob = candidate_labels[index][1]
            if label not in labels_to_id:
                labels_to_id[label] = len(labels_to_id)
            label_id = labels_to_id.get(label, len(labels_to_id))
            row_list.append({'article_id': article, 'label_id': label_id, 'prob': prob})

    return labels_to_id, pd.DataFrame(row_list)


def create_label_id_str_csv(directory, labels_to_ids):
    id_to_label = [(id, label) for (label, id) in labels_to_ids.items()]
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/lda_label_names.csv', index=False)


def create_article_label_csv(directory, article_label_df):
    article_label_df.to_csv(directory + "/article_lda_labels.csv", index=False)


def main(directory, article_topic_distribution_csv, topic_label_distribution_csv):
    article_topic_df = pd.read_csv(article_topic_distribution_csv)
    topic_label_df = pd.read_csv(topic_label_distribution_csv)

    topic_label_dic = create_topic_label_dic(topic_label_df)
    article_id_word_dic = {}

    for row in article_topic_df.itertuples():
        words = topic_label_dic[row.country]
        article_id_word_dic.setdefault(row.article_id, {})  # in case the word dic is empty

        for word, prob in words.items():
            if word not in article_id_word_dic[row.article_id]:
                article_id_word_dic[row.article_id].update({word: row.prob*prob})
            else:
                article_id_word_dic[row.article_id].update({word: article_id_word_dic[row.article_id][word] + row.prob*prob})

    labels_to_id, label_df = create_labels(article_id_word_dic)

    create_article_label_csv(directory, label_df)
    create_label_id_str_csv(directory, labels_to_id)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory, article_topic_distribution_csv, topic_label_distribution_csv = sys.argv[1:]
    main(directory, article_topic_distribution_csv, topic_label_distribution_csv)
