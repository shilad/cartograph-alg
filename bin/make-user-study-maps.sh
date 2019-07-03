#!/bin/bash
#
# Creates maps for all of the wikiprojects used in the Mechanical Turk user study
#
# Notes: Food was also used in this study, but the data for that wikiproject has already been collected.
#
# source ~/venv/cartograph-alg/bin/activate
#

set -e
set -x


projects=(Politics Technology Internet)


for i in {0..2}
do
    echo ${projects[$i]}
    python -m cartograph.data.domain_concept_creator data/${projects[$i]} ${projects[$i]} 5000
    python -m cartograph.data.label_creator_categories data/${projects[$i]}
    python -m cartograph.data.label_creator_keywords data/${projects[$i]}
    python -m cartograph.data.label_creator_hierarchical_cat data/${projects[$i]}
    python -m cartograph.data.label_creator_links data/${projects[$i]}
    python -m cartograph.data.vector_creator data/${projects[$i]} data/original_nav_vectors.txt.bz2
    python -m cartograph.data.popularity_creator data/${projects[$i]}
done