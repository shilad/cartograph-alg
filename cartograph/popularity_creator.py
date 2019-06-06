import pageviewapi as pv
import pandas as pd
import datetime
import sys
import os

def get_page_views(page_name):
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


def create_popularity(wikiproject):

    domain_concepts = pd.read_csv('/Users/research/Documents/Projects/cartograph-alg/data/' + wikiproject + '/domain_concept.csv')
    for index, row in domain_concepts.iterrows():
        try:
            pop = get_page_views(row['article_name'])
            pop_list.append(pop)
        except:
            pop_list.append('na')
    domain_concepts["popularity_score"] = pop_list
    return domain_concepts


def create_csv(wikiproject, directory):
    df = create_popularity(wikiproject)
    df.to_csv(directory + '/popularity_score.csv', index_label='article_id')


def main():
    if not os.path.exists(sys.argv[1]):
        os.mkdir(sys.argv[1])
    # articles = create_popularity(sys.argv[2])
    create_csv(sys.argv[2], sys.argv[1])


if __name__ == '__main__':
    main()