import pandas as pd
import numpy as np
import ast

def process(data_col_name, project_name):
    data = pd.read_csv(project_name + "_para_results.csv")
    print(data, len(data.index))
    col = data[data_col_name]
    # var = np.var(col)
    # print(var)
    # mean = np.mean(col)
    # print(mean)
    min = np.min(col)
    max = np.max(col)

    df = pd.DataFrame(None, index=sorted(data['label'].unique()), columns=data['low'].unique())

    for index, row in data.iterrows():

        df.at[float(row['low']), float(row['label'])] = (row[data_col_name] - min)/(max-min)
        if (row[data_col_name] == max):
            print(row)
            print("here")
            print(df.at[float(row['low']), float(row['label'])])
    df.to_csv(project_name + "_" + data_col_name + ".csv")
    # print(np.min(df))


# process("silhouette_score", "technology")
# process("mod", "technology")
# process("trustworthiness", "technology")
# process("ch_score", "technology")

def generateCSV(text_file):
    with open(text_file, "r") as file:
        rows = []
        str = file.read()
        curr, begin, end, length = 0, 0, 0, len(str)
        while curr < length:
            if str[curr] is '[': # beginning
                begin = curr
            elif str[curr] is ']':
                end = curr
            elif begin < end and str[curr-1] is not "\n":
                row = ast.literal_eval(str[begin : end + 1])
                rows += row
            curr += 1
        row = ast.literal_eval(str[begin: end + 1])
        rows += row
        df = pd.DataFrame(rows)
        df.to_csv("/home/rockpang/Desktop/cartograph-alg/cartograph/params/food_para_results.csv", index=False)
        print(df)


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
process("low_dist", "food")
process("label_dist", "food")
process("high_dist", "food")
#
# generateCSV("source.txt")