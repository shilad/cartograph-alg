"""
Given a category and number of desired articles, output a list of article ids and titles
that belong to that category.

This implementation uses WikiProjects and the enwp10 tool to generate the list.

Author: Lily Irvin
"""


from bs4 import BeautifulSoup
from requests import get
import os
import pandas as pd
import re


def create_urls(wikiproject, num_articles):
    urls = []
    offset = 1
    while offset < int(num_articles):
        urls.append('https://tools.wmflabs.org/enwp10/cgi-bin/list2.fcgi?run=yes&projecta=' + wikiproject +
                    '&namespace=&pagename=&quality=&importance=&score=&limit=1000&offset=' + str(offset) +
                    '&sorta=Quality&sortb=Quality')
        offset += 1000
    return urls[:num_articles]


def create_domain_concept(urls, num_articles):
    articles = []
    article_count = 0
    STOP_WORDS = {'list', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september',
                  'october', 'november', 'december'}
    for url in urls:
        response = get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        center = soup.find('center')
        rows = center.find_all('tr')
        for i in range(1, len(rows)):
            if article_count == num_articles:
                break
            article = rows[i].find_all('td')
            title = article[1].a.text.strip()
            words = title.lower().split()
            words_unique = set()
            for word in words:
                words_unique.add(word)
            if not re.match(".*[0-9]{4} .*", title) and not re.match(".*[0-9]+$", title) and not \
                    re.match(".*Category.*", title) and words_unique.intersection(STOP_WORDS) == set():
                articles.append(article[1].a.text.strip())
            else:
                print(article[1].a.text.strip())
            article_count += 1
    return articles


def create_csv(articles, directory):
    df = pd.DataFrame(articles, columns=['article_name'])
    df.to_csv(directory + '/domain_concept.csv', index_label='article_id')


def main(map_directory, project_name, number_of_articles):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    urls = create_urls(project_name, number_of_articles)
    articles = create_domain_concept(urls, number_of_articles)
    create_csv(articles, map_directory)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 4:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory, project_name, number_of_articles = sys.argv[1:]
    main(map_directory, project_name, int(number_of_articles))