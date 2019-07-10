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


projects=(Technology Internet)
directories=(technology internet)


for i in {0..1}
do
    echo ${projects[$i]}
    python -m cartograph.data.domain_concept_creator data/${directories[$i]} ${projects[$i]} 5000
    python -m cartograph.data.label_creator_categories data/${directories[$i]}
    python -m cartograph.data.label_creator_hierarchical_cat data/${directories[$i]}
    python -m cartograph.data.label_creator_links data/${directories[$i]}
    python -m cartograph.data.text_getter data/${directories[$i]}
    python -m cartograph.data.label_creator_keywords data/${directories[$i]}
    python -m cartograph.data.label_creator_key-phrases data/${directories[$i]}
    python -m cartograph.data.vector_creator data/${directories[$i]} data/original_nav_vectors.txt.bz2
    python -m cartograph.data.popularity_creator data/${directories[$i]}
done