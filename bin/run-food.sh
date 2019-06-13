#!/bin/bash
#
# Example shell script to run the food pipeline.
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/run-food.sh

# python -m cartograph.domain_concept_creator data/food Food_and_drink 5000
# python -m cartograph.label_creator data/food
python -m cartograph.vector_creator data/food data/original_vectors combined
# python -m cartograph.popularity_creator data/food
# python -m cartograph.xy_embeddings_builder data/food
# python -m cartograph.evaluation data/food/
python -m cartograph.cluster_builder_kmeans data/food original
# python -m cartograph.label_selector data/food

# python -m cartograph.json_generator data/food/
# python -m cartograph.svg_generator data/food 1500 1500 colorblind

