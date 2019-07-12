#!/bin/bash
#
# Example shell script to create a map dataset
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Note: this script does not call domain_concept_creator.py because the domain concept was scraped from a different
# location (Shilad's friend Aaron's file from WikiMedia)
#

set -e

python -m cartograph.data.label_creator_categories data/culture
python -m cartograph.data.label_creator_keywords data/culture
python -m cartograph.data.label_creator_hierarchical_cat data/culture
python -m cartograph.data.label_creator_links data/culture
python -m cartograph.data.label_creator_combined data/culture
python -m cartograph.data.vector_creator data/culture data/original_nav_vectors.txt.bz2
python -m cartograph.data.popularity_creator data/culture