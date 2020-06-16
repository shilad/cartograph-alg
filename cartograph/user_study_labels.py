import pandas as pd
pd.options.display.max_columns = 20
import pandas as pd

"""
This script creates the user study candidate labels for each experiment. 
"""


def union_label_candidates(path, k):
    check = pd.read_csv(path + "/check.csv")
    check['complex'] = check['tf'] * check['idf'] * check['sum']
    check['simple'] = check['tf'] * check['idf']
    check = check[['country', 'new_name', 'simple', 'complex']].drop_duplicates()
    simple = check.sort_values('simple', ascending=False).groupby("country").head(k).sort_values('country')
    complex = check.sort_values('complex', ascending=False).groupby("country").head(k).sort_values('country')
    union = pd.concat([simple, complex]).drop_duplicates().sort_values('country').reset_index().drop(columns=['index'])
    union.to_csv(path + "/union.csv")
    return union


union = union_label_candidates("../experiments/technology/0216", 10)

print(union)

