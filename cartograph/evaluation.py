"""
Given a list of vector representations of Wikipedia articles and 2D representations of them
output a value in [0,1] measuring how much of the local structure is maintained in lower dimensions.
0 being the worst and 1 being the best.
The metric utilized is known as trustworthiness. Described in detail by

    * "Neighborhood Preservation in Nonlinear Projection Methods: An
      Experimental Study"
      J. Venna, S. Kaski
    * "Learning a Parametric Embedding by Preserving Local Structure"
      L.J.P. van der Maaten
We plan to add other evaluation metrics
Author: Jonathan Scott
"""

import pandas as pd
from sklearn.manifold.t_sne import trustworthiness


def evaluate_xy_embed(x, x_embed):
    return trustworthiness(x, x_embed)


def main(map_directory):
    X = pd.read_csv(map_directory+"article_vectors.csv").iloc[:, 2:]
    X_embed = pd.read_csv(map_directory+"xy_embeddings.csv").iloc[:, 1:]
    print("Trustworthiness: " +str(evaluate_xy_embed(X, X_embed)))


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)