import pandas as pd
import re
from collections import defaultdict


VALIDATION_ARTICLES = ['Naruto', 'Gmail', 'Urdu', 'Mathematical Statistics', 'Computer Science', 'Blush', 'Painting', 'Earbuds', 'Braces', 'Hairstyle', 'Tamarind', 'Diapers', 'Baby Powder', 'Lmao', 'Satellite', 'Quiz', 'Vanilla', 'Mistake', 'Four-leaf clover', 'Mac n\' Cheetos', 'Bleach', 'Aroma of Tacoma', 'Cowboy', 'Birthday Cake', 'The Moon is made of Green Cheese', 'Vampire', '1896 Summer Olympics', 'Caribbean', 'Beach', 'Ramen', 'Braces', 'Chocolate', 'American Revolutionary War', 'Serum', 'Old Town Road', 'Sailor Moon', 'Limbo', 'The Lion King', 'Braces', 'Necklace', 'Abdomen', 'Bumblebee']


def validation_metric(responses):
    """
    Input: Turker responses
    Output: Number of dummy articles caught by turkers, number of total dummy articles in HIT's, and a dictionary of
            missed dummy articles (key = article name, value = number of times it was missed)
    """
    num_validated = 0
    missed = defaultdict(int)
    for index, row in responses.iterrows():
        for column in responses.columns:
            if 'Answer.dont-belong' in column:
                article_num = re.findall("(?<=_)(.*)(?=\.)", column)[0]
                if row['Input.article_' + str(article_num)] in VALIDATION_ARTICLES:
                    if row[column]:
                        num_validated += 1
                    else:
                        missed[row['Input.article_' + str(article_num)]] += 1

    total_possible = responses.shape[0] * 4
    return num_validated, total_possible, missed


def main(mturk, responses):
    num_validated, total_possible, missed = validation_metric(responses)
    print(missed)
    print('number of dummy articles validated: ' + str(num_validated))
    print('number of total dummy articles: ' + str(total_possible))


responses = pd.read_csv('study/evaluation/sample_batch.csv')
mturk = pd.read_csv('study/mturk.csv')
main(mturk, responses)
