#!/bin/bash
#
# Example shell script to create a map dataset
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Note that this probably doesn't work. Please get it working!
#

set -e

#python -m cartograph.data.domain_concept_creator data/food Food_and_drink 5000
#python -m cartograph.data.label_creator_categories data/food
#python -m cartograph.data.vector_creator data/food data/original_nav_vectors.txt.bz2
#python -m cartograph.data.popularity_creator data/food
python -m cartograph.data.label_creator_keywords data/food

