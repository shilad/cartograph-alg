import pandas as pd
import numpy as np
trustworthiness = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0076/trustworthiness_evaluation.json", lines=True)
MI_score = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0076/cluster_evaluation.json", lines=True)
modularity = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0076/mod_evaluation.json", lines=True)


trustworthiness = np.array(trustworthiness).reshape(int(len(trustworthiness) / 3), 3)
MI_score = np.array(MI_score).reshape(int(len(MI_score) / 2), 2)
modularity = np.array(modularity).reshape(int(len(modularity) / 3), 3)

trustworthiness_df = pd.DataFrame(trustworthiness)
MI_score_df = pd.DataFrame(MI_score)
modularity_df = pd.DataFrame(modularity)

pd.set_option('display.max_columns', 30)


trustworthiness_df.columns = ['orig_trust', 'local_trust', 'low_trust']
MI_score_df.columns = ['local_MI', 'low_MI']
modularity_df.columns = ['orig_mod', 'local_mod', 'low_mod']

x = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
index = np.repeat(x, 5)

print(trustworthiness_df.shape)
print(modularity_df.shape)
print(MI_score_df.shape)

trustworthiness_df = trustworthiness_df.join(pd.DataFrame(index)).groupby(0).mean()
modularity_df = modularity_df.join(pd.DataFrame(index)).groupby(0).mean()
MI_score_df = MI_score_df.join(pd.DataFrame(index)).groupby(0).mean()
# final.set_index(index[0:7])
print(trustworthiness_df)
print(modularity_df)
print(MI_score_df)
trustworthiness_df.to_csv('/Users/research/desktop/trust.csv')
modularity_df.to_csv('/Users/research/desktop/mod.csv')
MI_score_df.to_csv('/Users/research/desktop/MI.csv')