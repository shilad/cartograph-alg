import wikipediaapi
import os
import pandas as pd


def scrape_article(articles_csv):
    df = pd.read_csv(articles_csv)
    wiki_pages = {}
    for index, row in df.iterrows():
        wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
        page = wiki_wiki.page(row['article_name'])
        wiki_pages[row['article_id']] = page.text
    return wiki_pages


def create_csv(map_directory, article_text):
    df = pd.DataFrame.from_dict(article_text, orient='index', columns=['article_text'])
    df.index.name = 'article_id'
    df.to_csv(map_directory + '/article_text.csv')


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    text = scrape_article(map_directory + '/domain_concept.csv')
    create_csv(map_directory, text)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)

