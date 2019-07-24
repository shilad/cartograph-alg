import pandas as pd
import re


VALIDATION_ARTICLES = ['Naruto', 'Gmail', 'Urdu', 'Mathematical Statistics', 'Computer Science', 'Blush', 'Painting',
                       'Earbuds', 'Braces', 'Hairstyle', 'Tamarind', 'Diapers', 'Baby Powder', 'Lmao', 'Satellite',
                       'Quiz', 'Vanilla', 'Mistake', 'Four-leaf clover', 'Mac n\' Cheetos', 'Bleach', 'Aroma of Tacoma',
                       'Cowboy', 'Birthday Cake', 'The Moon is made of Green Cheese', 'Vampire', '1896 Summer Olympics',
                       'Caribbean', 'Beach', 'Ramen', 'Braces', 'Chocolate', 'American Revolutionary War', 'Serum',
                       'Old Town Road', 'Sailor Moon', 'Limbo', 'The Lion King', 'Braces', 'Necklace', 'Abdomen',
                       'Bumblebee']


def check_workers(responses):
    for worker in responses['WorkerId'].unique():
        worker_responses = responses.loc[responses['WorkerId'] == worker]
        if not validation_articles(worker_responses):
            responses = responses[responses['WorkerId'] != worker]
    return responses


def validation_articles(worker_responses):
    """
    Input: Turker responses
    Output: Whether or not the turker's responses should be kept, number of dummy articles caught by turkers,
            and number of total dummy articles in HIT's
    """
    num_validated = 0
    num_discarded = 0
    for index, row in worker_responses.iterrows():
        for column in worker_responses.columns:
            if 'Answer.dont-belong' in column:
                if row[column]:
                    num_discarded += 1
                    article_num = re.findall("(?<=_)(.*)(?=\.)", column)[0]
                    if row['Input.article_' + str(article_num)] in VALIDATION_ARTICLES:
                        num_validated += 1

    total_possible = worker_responses.shape[0] * 4
    total_articles = worker_responses.shape[0] * 120
    return num_discarded / total_articles <= 0.5 <= num_validated / total_possible


def main(responses):
    cleaned_responses = check_workers(responses)

    # print(cleaned_responses.shape)
    # print(cleaned_responses)

    cleaned_responses.to_csv('study/evaluation/cleaned_mturk_results.csv')


responses = pd.read_csv('study/evaluation/mturk_results.csv')
main(responses)
