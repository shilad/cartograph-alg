#!/bin/bash
#
# Example shell script to create a map dataset
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#

set -e

python -m cartograph.domain_concept_creator data/food Food_and_drink 5000
python -m cartograph.label_creator data/food
python -m cartograph.vector_creator data/food data/original_vectors
python -m cartograph.popularity_creator data/food

