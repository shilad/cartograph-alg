#!/usr/bin/env python

import logging
from collections import defaultdict

import pandas as pd
import re
import sys


def normalize(article):
    return article.lower().replace(' ', '_')

VALIDATION_ARTICLES = {
    normalize(a) for a in
    ['Naruto', 'Gmail', 'Urdu', 'Mathematical Statistics', 'Computer Science', 'Blush', 'Painting',
     'Earbuds', 'Braces', 'Hairstyle', 'Tamarind', 'Diapers', 'Baby Powder', 'Lmao', 'Satellite',
     'Quiz', 'Vanilla', 'Mistake', 'Four-leaf clover', 'Mac n\' Cheetos', 'Bleach', 'Aroma of Tacoma',
     'Cowboy', 'Birthday Cake', 'The Moon is made of Green Cheese', 'Vampire', '1896 Summer Olympics',
     'Caribbean', 'Beach', 'Ramen', 'Braces', 'Chocolate', 'American Revolutionary War', 'Serum',
     'Old Town Road', 'Sailor Moon', 'Limbo', 'The Lion King', 'Braces', 'Necklace', 'Abdomen',
     'Bumblebee']
}

INVALID_LABELS = {normalize(l) for l in ['internet', 'media', 'food', 'technology']}

class ClusterResponse(object):
    def __init__(self, worker, hit, project, cluster_alg, cluster_num):
        self.worker = worker
        self.hit = hit
        self.project = project
        self.cluster_alg = cluster_alg
        self.cluster_num = cluster_num
        self.labels = []
        self.chosen_labels = []
        self.articles = []
        self.flagged_articles = set()
        self.num_valid_seen = 0
        self.num_valid_flagged = 0


def parse_and_filter(path='mturk_results.csv'):
    responses = parse_hits(path)
    filtered = filter_bots(responses)
    logging.info('filtered from %d to %d responses', len(responses), len(filtered))
    return filtered


def parse_hits(path='mturk_results.csv'):
    MAT_GROUP = re.compile('(.*)_c(\d+)_g(\d+)').match
    responses = []

    logging.info('parsing HITs from %s', path)
    for i, (row_id, row) in enumerate(pd.read_csv(path).iterrows()):
        hit = row.HITId
        worker = row.WorkerId
        for p in range(4):
            mt_group = row['Input.group_id_' + str(p)]
            m = MAT_GROUP(mt_group)
            cluster_alg = m.group(1)
            cluster_num = m.group(2)
            mt_group_id = m.group(3)
            proj = row['Input.project_' + str(p)]
            country  = row['Input.country_' + str(p)]
            assert(int(cluster_num) == int(country))
            resp = ClusterResponse(worker, hit, proj, cluster_alg, cluster_num)

            for a in range(30):
                article = row['Input.article_%d_%d' % (p, a)]
                if type(article) is str: # not NaN
                    article = normalize(article)
                    flagged = bool(row['Answer.dont-belong_%d_%d.%d' % (p, a, a)])
                    if article in VALIDATION_ARTICLES:
                        resp.num_valid_seen += 1
                        if flagged: resp.num_valid_flagged += 1
                    else:
                        resp.articles.append(article)
                        if flagged:
                            resp.flagged_articles.add(article)

            for l in range(25):
                label = row['Input.label_%d_%d' % (p, l)]
                if type(label) is str: # not NAN
                    resp.labels.append(normalize(label))
                else:
                    resp.labels.append(None)

            for c in range(1, 6):
                c = row['Answer.chosen_label_%d_%d' % (p, c)]
                if c >= 0:
                    resp.chosen_labels.append(resp.labels[c])

            resp.labels = [l for l in resp.labels if l is not None] # filter out NAs
            responses.append(resp)

    logging.info('found %d cluster responses for %d hits and %d workers in %s',
                 len(responses), len({ r.hit for r in responses}), len({ r.worker for r in responses}), path)

    return responses


def is_valid_worker(worker_responses):
    num_articles = 0
    num_articles_flagged = 0
    num_valid_seen = 0
    num_valid_flagged = 0
    for r in worker_responses:
        num_articles += len(r.articles)
        num_articles_flagged += len(r.flagged_articles)
        num_valid_seen += r.num_valid_seen
        num_valid_flagged += r.num_valid_flagged
    fraction_valid = 1.0 * num_valid_flagged / num_valid_seen
    fraction_flagged = 1.0 * num_articles_flagged / num_articles

    return fraction_valid >= 0.5 and fraction_flagged <= 0.6


def filter_bots(responses):
    by_worker = defaultdict(list)
    for r in responses:
        by_worker[r.worker].append(r)
    filtered = []
    for w, w_resps in by_worker.items():
        if is_valid_worker(w_resps):
            filtered.extend(w_resps)
    return filtered


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    responses = parse_and_filter()
