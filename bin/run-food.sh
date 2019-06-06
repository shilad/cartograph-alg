#!/bin/bash
#
# Example shell script to run the food pipeline.
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/run-food.sh

python -m cartograph.domain_concept_creator data/food Food_and_drink 5000
# Add label_creator command here
# Add vector_creator command here
# Add popularity_creator command here
python -m cartograph.popularity_creator data/food food

python -m cartograph.xy_embeddings_builder data/food article_vectors.csv