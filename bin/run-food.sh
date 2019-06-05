#!/bin/bash
#
# Example shell script to run the food pipeline.
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/run-food.sh

python -m cartograph.domain_concept_creator data/food Food_and_drink 5000