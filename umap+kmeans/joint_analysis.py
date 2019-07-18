import pandas as pd
import numpy as np
trustworthiness = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0101/trustworthiness_evaluation.json", lines=True)
MI_score = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0101/cluster_evaluation.json", lines=True)
modularity = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0101/mod_evaluation.json", lines=True)
centroid = pd.read_json("/Users/research/Documents/Projects/cartograph-alg/experiments/food/0101/centroid_evaluation.json", lines=True)


trustworthiness = np.array(trustworthiness).reshape(int(len(trustworthiness) / 3), 3)
MI_score = np.array(MI_score).reshape(int(len(MI_score) / 6), 6)
modularity = np.array(modularity).reshape(int(len(modularity) / 3), 3)
centroid = np.array(centroid).reshape(int(len(centroid) / 3), 3)

trustworthiness_df = pd.DataFrame(trustworthiness)
MI_score_df = pd.DataFrame(MI_score)
modularity_df = pd.DataFrame(modularity)
centroid = pd.DataFrame(centroid)
pd.set_option('display.max_columns', 30)


trustworthiness_df.columns = ['orig_trust', 'local_trust', 'low_trust']
MI_score_df.columns = ['orig_sil', 'orig_ch', 'local_sil', 'local_ch', 'low_sil', 'low_ch']
modularity_df.columns = ['orig_mod', 'local_mod', 'low_mod']
centroid.columns = ['orig_centroid', 'local_centroid', 'low_centroid']

x = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
index = np.repeat(x, 10)


assert(trustworthiness_df.shape[0] == modularity_df.shape[0] == MI_score_df.shape[0] == centroid.shape[0])


trustworthiness_df = trustworthiness_df.join(pd.DataFrame(index)).groupby(0).mean()
modularity_df = modularity_df.join(pd.DataFrame(index)).groupby(0).mean()
MI_score_df = MI_score_df.join(pd.DataFrame(index)).groupby(0).mean()
centroid = centroid.join(pd.DataFrame(index)).groupby(0).mean()

# print(trustworthiness_df)
# print(modularity_df)
# print(MI_score_df)
# print(centroid)

# trustworthiness_df.to_csv('/Users/research/desktop/trust6.csv')
# modularity_df.to_csv('/Users/research/desktop/mod6.csv')
# MI_score_df.to_csv('/Users/research/desktop/MI6.csv')
# centroid.to_csv('/Users/research/desktop/centroid6.csv')


