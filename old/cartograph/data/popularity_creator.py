"""
Given a domain concept, gets the popularity score for each article in that domain concept,
produces a csv file that contains article ids and its corresponding popularity scores.
This implementation uses Wikipedia's PageView API.
Author: Lu Li
"""
import pageviewapi as pv
import pandas as pd
import datetime
import os


def get_popularity_score(page_name):
    """Get number of views of a page for a specific food using PageView API from Wikipedia."""
    today = datetime.date.today().strftime('%Y%m%d')
    last_year_today = (datetime.date.today() - datetime.timedelta(days=365)).strftime('%Y%m%d')
    page_views = pv.per_article(page=page_name, project='en.wikipedia', start=last_year_today, end=today,
                                granularity='monthly').get('items')
    views = 0
    for i in range(len(page_views)):
        views += page_views[i].get('views')
    return views


pop_list = []


def create_popularity_score(path):

    domain_concepts = pd.read_csv(path + '/domain_concept.csv')
    x = 0
    for index, row in domain_concepts.iterrows():
        if x % 1000 == 0:
            print(str(x) + " articles completed")
        try:
            popularity_score = get_popularity_score(row['article_name'])
            pop_list.append(popularity_score)
            x += 1
        except:
            pop_list.append('na')
            x += 1
    domain_concepts["popularity_score"] = pop_list
    domain_concepts = domain_concepts[domain_concepts['popularity_score'] != 'na']
    return domain_concepts.drop('article_name', axis=1).iloc[:, 1:]


def create_csv(directory):
    df = create_popularity_score(directory)
    df.to_csv(directory + '/popularity_score.csv', index_label='article_id')


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    create_csv( map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
