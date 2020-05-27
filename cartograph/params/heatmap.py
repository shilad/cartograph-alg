from scipy.ndimage import convolve

import seaborn as sns
import pandas as pd
from matplotlib import pyplot



def sum_mean(project):

    low = pd.read_csv("./" + project + "_" + "low_dist" + ".csv")
    high = pd.read_csv("./" + project + "_" + "high_dist" + ".csv")
    label = pd.read_csv("./" + project + "_" + "label_dist" + ".csv")
    sum = low + high + label
    return sum

def generate_heat_map(project, weight):
    pyplot.figure(figsize=(20, 15))
    data = pd.read_csv("./" + project + "_" + weight + ".csv")
    x = data.iloc[:, 0]
    data = data.iloc[:, 1:]
    data = data.set_index(x)
    heatmap = sns.heatmap(data,
                          cmap="Blues",
                          square=True,
                          annot=True,
                          fmt=".3f",
                          annot_kws={'size': 10},
                          linecolor="white",
                          linewidths=0.3,
                          cbar_kws={"shrink": .8}
    )

    heatmap.set_ylabel('Low dimensional distance weight')
    heatmap.set_xlabel('Label dimensional distance weight')
    heatmap.set_title(project + " mean " + weight)

    heatmap.get_figure().savefig("./vis/" + project + "_" + weight + ".png")


def generate_sum_heat_map(project):
    pyplot.figure(figsize=(20, 15))
    data = sum_mean(project)
    x = round(data.iloc[:, 0]/3, ndigits=2)
    data = data.iloc[:, 1:]
    data = data.set_index(x)
    h_diag = [[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]
    data = convolve(data, h_diag, mode='constant')
    data = pd.DataFrame(data)
    data = data.set_index(x)
    data.columns = x
    heatmap = sns.heatmap(data,
                          cmap="Blues",
                          square=True,
                          annot=True,
                          fmt=".3f",
                          annot_kws={'size': 10},
                          linecolor="white",
                          linewidths=0.3,
                          cbar_kws={"shrink": .8}
    )

    heatmap.set_ylabel('Low dimensional distance weight')
    heatmap.set_xlabel('Label dimensional distance weight')
    heatmap.set_title(project + " sum mean distance")

    heatmap.get_figure().savefig("./vis/" + project + "_" + "sum" + ".png")

weights = ["low_dist", "high_dist", "label_dist"]

projects = ["food"]

# project = "internet"
for project in projects:
    for weight in weights:
        generate_heat_map(project, weight)
    generate_sum_heat_map(project)

# generate_sum_heat_map("geography")
# print(sum_mean("geography"))
