import csv
from collections import defaultdict
import pandas as pd
import time

vectors = {}

def to_float(string):
    try:
        return float(string)
    except ValueError:
        pass

start = time.time()
with open('../data/original_vectors', encoding="ISO-8859-1") as file:
    for line in file:
        values = line.split()
        vectors[values[0]] = [to_float(x) for x in values[1:]]
end = time.time()
print("Reading in data takes: "+str(end-start)+" seconds.")

domain_concept_df = pd.read_csv('../data/food/domain_concept.csv')
temp_map_dict = {}# for fast look up
for i, row in domain_concept_df.iterrows():
    temp_map_dict[row['article_name'].replace(" ", "_")] = row['article_id']
#print()

# with open('../data/food/domain_concept.csv', encoding="ISO-8859-1") as data:
#     articles = [row[0] for row in csv.reader(data, delimiter='\n')]
#     articles[:] = [x.replace(" ", "_") for x in articles]

article_set = set(temp_map_dict.keys())# O(1) look up.
article_vectors_df_ready = []
for food_article in vectors.keys():
    if food_article in article_set:
        article_vectors_df_ready.append([food_article]+vectors[food_article])

#print(len(article_vectors_df_ready))

vector_ids = ['vector_'+str(i) for i in range(100)]# we know the size of the vectors previously

domain_concept_df = pd.read_csv('../data/food/domain_concept.csv')
temp_map_dict = {}# for fast look up
for i,row in domain_concept_df.iterrows():
    temp_map_dict[row['article_name'].replace(" ", "_")] = row['article_id']

# a Dataframe expects a dictionary where {col:[list of values in column]
article_w_vectors = pd.DataFrame(article_vectors_df_ready, columns=['article_name']+vector_ids)
for i,row in article_w_vectors.iterrows():
    #article_w_vectors.insert(0, 'article_id', temp_map_dict[row['article_name']])
    article_w_vectors.loc[i, 'article_id'] = temp_map_dict[row['article_name']]

print(article_w_vectors.head())
article_w_vectors.to_csv("../data/article_vectors.csv") # get article ids from domain_concepts file








