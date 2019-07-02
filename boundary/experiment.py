import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt


df = pd.read_csv('../experiments/food/0009/xy_embeddings.csv')

points = np.zeros(shape=(df.shape[0], 2))


for index, row in df.iterrows():
    points[index] = [row['x'], row['y']]

plt.plot(points[:, 0], points[:, 1], 'o', color='black')
vor = Voronoi(points)



print(vor)
voronoi_plot_2d(vor)
plt.show()


def relax_points(times=2):
    for i in range(times):
        centroids = []
        for region in