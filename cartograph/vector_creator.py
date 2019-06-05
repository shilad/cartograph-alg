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
with open('../data/original-vectors', encoding="ISO-8859-1") as file:
    for line in file:
        values = line.split()
        vectors[values[0]] = [to_float(x) for x in values[1:]]
end = time.time()
print("Reading in data takes: "+str(end-start)+" seconds.")

with open('../data/data.csv', encoding="ISO-8859-1") as data:
    articles = [row[0] for row in csv.reader(data, delimiter='\n')]
    articles[:] = [x.replace(" ", "_") for x in articles]

article_set = set(articles)# O(1) look up.
article_vectors_df_ready = []
for food_article in vectors.keys():
    if food_article in article_set:
        article_vectors_df_ready.append([food_article]+vectors[food_article])

#print(food_vectors_df_ready[:10])
print(len(article_vectors_df_ready))

vector_ids = ['vector_'+str(i) for i in range(100)]# we know the size of the vectors previously
#food_vectors_df_ready = [[i[0]]+i[1] for i in food_vectors.items()]

# a Dataframe expects a dictionary where {col:[list of values in column]
article_w_vectors = pd.DataFrame(article_vectors_df_ready,columns=['article_name']+vector_ids)
print(article_w_vectors.head())
article_w_vectors.to_csv("../data/article_vectors.csv",index_label='article_id')








