import wikipediaapi
import os
import pandas as pd


def scrape_links(articles_csv):
    df = pd.read_csv(articles_csv)
    wiki_links = {}
    for index, row in df.iterrows():
        wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)
        page = wiki_wiki.page(row['article_name'])
        links = page.links
        wiki_links[page.title] = list(links.keys())
    return wiki_links


def create_csv(map_directory, links):
    df = pd.DataFrame.from_dict(links, orient='index')
    df.index.name = 'article_id'
    df.to_csv(map_directory + '/article_links.csv')


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)
    links = scrape_links(map_directory + '/domain_concept.csv')
    create_csv(map_directory, links)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory project_name number_of_articles' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)

