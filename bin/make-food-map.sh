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

python -m cartograph.inputs.domain_concept_creator data/food Food_and_drink 5000
python -m cartograph.inputs.label_creator data/food
python -m cartograph.inputs.vector_creator data/food data/original_vectors
python -m cartograph.inputs.popularity_creator data/food

