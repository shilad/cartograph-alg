# TODO: consider other labels.
#

from collections import defaultdict
import pandas as pd

import cluster_responses


class ArticleInfo(object):
    def __init__(self, name):
        self.name = name
        self.num_exposures = 0
        self.num_flags = 0

class LabelInfo(object):
    def __init__(self, name):
        self.name = name
        self.num_exposures = 0
        self.ranks = []

def main():
    # Calculate labels and articles using ALL turker responses
    group_articles = defaultdict(dict)
    group_labels = defaultdict(dict)
    for r in cluster_responses.parse_hits():
        group = (r.project, r.cluster_alg, r.cluster_num)
        for a in r.articles:
            group_articles[group][a] = ArticleInfo(a)
        for l in r.labels:
            group_labels[group][l] = LabelInfo(l)

    # Calculate user feedback using valid responses
    for r in cluster_responses.parse_and_filter():
        group = (r.project, r.cluster_alg, r.cluster_num)
        for a in r.articles:
            group_articles[group][a].num_exposures += 1
            if a in r.flagged_articles:
                group_articles[group][a].num_flags += 1
        for l in r.labels:
            group_labels[group][l].num_exposures += 1
            if l in r.chosen_labels:
                group_labels[group][l].ranks.append(r.chosen_labels.index(l))

    # Write out articles
    rows = []
    for ((proj, cluster_alg, cluster_num), article_info) in group_articles.items():
        for ai in article_info.values():
            rows.append({
                'project' : proj,
                'cluster_alg' : cluster_alg,
                'cluster_num' : cluster_num,
                'name' : ai.name,
                'exposures' : ai.num_exposures,
                'flags' : ai.num_flags
            })

    df = pd.DataFrame(rows,
                      columns=['project', 'cluster_alg', 'cluster_num', 'name', 'exposures', 'flags'])
    df.to_csv('hit_articles.csv')

    # Write out labels
    rows = []
    for ((proj, cluster_alg, cluster_num), label_info) in group_labels.items():
        for li in label_info.values():
            rows.append({
                'project' : proj,
                'cluster_alg' : cluster_alg,
                'cluster_num' : cluster_num,
                'name' : li.name,
                'exposures' : li.num_exposures,
                'count' : len(li.ranks),
                'share' : 1.0 * len(li.ranks) / li.num_exposures,
                'borda' : sum([5 - r for r in li.ranks]),
                'avg_borda': 1.0 * sum([5 - r for r in li.ranks]) / li.num_exposures
            })

    df = pd.DataFrame(rows,
                      columns=['project', 'cluster_alg', 'cluster_num', 'name',
                               'exposures', 'count', 'share', 'borda', 'avg_borda'])
    df.to_csv('hit_labels.csv')

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
