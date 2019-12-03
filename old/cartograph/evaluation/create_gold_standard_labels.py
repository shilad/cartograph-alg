import json
import pandas as pd


def load_json(file_path):
    articles = []
    for line in open(file_path, 'r'):
        articles.append(json.loads(line))
    return articles


def filter_articles(category, data):
    articles = {}
    for article in data:
        if len(article['mid_level_categories']) > 0:

            if article['mid_level_categories'][0].startswith(category):
                articles[article['talk_page_title']] = {'label': article['mid_level_categories'][0].replace(category +
                                                                                                            '.', ''),
                                                        'category': category,
                                                        'projects': article['templates']}
    return articles


def generate_category_articles_csv(articles):
    df = pd.DataFrame.from_dict(articles, orient='index')
    df.to_csv('../data/gold_standard_labels.csv', index_label='article')


def main():
    data = load_json('../data/enwiki.labeled_wikiprojects.json')
    articles = filter_articles('Culture', data)
    generate_category_articles_csv(articles)


if __name__ == '__main__':
    main()
