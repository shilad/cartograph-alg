#!/bin/bash
#
# Example shell script to run the food pipeline.
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/run-food_experiments.sh

# python -m cartograph_pop_svd_experiment.domain_concept_creator data/food Food_and_drink 5000
# python -m cartograph_pop_svd_experiment.label_creator data/food
# python -m cartograph_pop_svd_experiment.vector_creator data/food data/original_vectors
# python -m cartograph_pop_svd_experiment.popularity_creator data/food
# python -m cartograph_pop_svd_experiment.xy_embeddings_builder data/food
# python -m cartograph_pop_svd_experiment.evaluation data/food/
# python -m cartograph_pop_svd_experiment.cluster_builder_kmeans data/food original
python -m cartograph_pop_svd_experiment.label_selector data/food svd_pop

# python -m cartograph_pop_svd_experiment.json_generator data/food/
# python -m cartograph_pop_svd_experiment.svg_generator data/food 1500 1500 colorblind

