from bs4 import BeautifulSoup
from requests import get
import sys
import os
import pandas as pd


def create_urls(wikiproject, num_articles):
    urls = []
    offset = 1
    while offset < int(num_articles):
        urls.append('https://tools.wmflabs.org/enwp10/cgi-bin/list2.fcgi?run=yes&projecta=' + wikiproject + '&namespace=&pagename=&quality=&importance=&score=&limit=1000&' + str(offset) + '&sorta=Quality&sortb=Quality')
        offset += 1000
    return urls


def create_domain_concept(urls):
    articles = []
    for url in urls:
        response = get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        center = soup.find('center')
        rows = center.find_all('tr')
        for i in range(1, 1001):
            article = rows[i].find_all('td')
            articles.append(article[1].a.text.strip())
    return articles


def create_csv(articles, directory):
    df = pd.DataFrame(articles, columns=['article_name'])
    df.to_csv(directory + '/domain-concept.csv', index_label='article_id')


def main():
    if not os.path.exists('../' + sys.argv[1]):
        os.mkdir('../' + sys.argv[1])
    urls = create_urls(sys.argv[2], sys.argv[3])
    articles = create_domain_concept(urls)
    create_csv(articles, '../' + sys.argv[1])


if __name__ == '__main__':
    main()
