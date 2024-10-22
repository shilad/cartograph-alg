#!/bin/bash
#
# Creates maps for all of the wikiprojects used in the Mechanical Turk user study
#
# source ~/venv/cartograph-alg/bin/activate
#

set -e
set -x


projects=(Food_and_drink Internet Media Technology)
directories=(food internet media technology)


for i in {2..2}
do
    echo ${projects[$i]}
    python -m cartograph.data.domain_concept_creator data/${directories[$i]} ${projects[$i]} 5000
    python -m cartograph.data.label_creator_hierarchical_cat data/${directories[$i]}
    python -m cartograph.data.label_creator_links data/${directories[$i]}
    python -m cartograph.data.text_getter data/${directories[$i]}
    python -m cartograph.data.label_creator_keywords data/${directories[$i]}
    python -m cartograph.data.label_creator_key_phrases data/${directories[$i]}
    python -m cartograph.data.label_creator_combined data/${directories[$i]}
    python -m cartograph.data.vector_creator data/${directories[$i]} data/original_nav_vectors.txt.bz2
    python -m cartograph.data.popularity_creator data/${directories[$i]}
done
