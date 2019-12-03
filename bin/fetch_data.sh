#!/bin/bash
#
# fetch necessary raw data to feed in algorithms

set -e
set -x

projects=(Education Geography)
directories=(education georgraphy)
num_of_articles=(93811 353772)     # need to find out the number of articles within the topic.


for i in {0..2}
do
    echo ${projects[$i]}
    python -m cartograph.data.domain_concept_creator data/${directories[$i]} ${projects[$i]} ${num_of_articles[$i]}
    python -m cartograph.data.label_creator_hierarchical_cat data/${directories[$i]}
    python -m cartograph.data.label_creator_links data/${directories[$i]}
    python -m cartograph.data.text_getter data/${directories[$i]}
    python -m cartograph.data.label_creator_keywords data/${directories[$i]}
    python -m cartograph.data.label_creator_key_phrases data/${directories[$i]}
    python -m cartograph.data.label_creator_combined data/${directories[$i]}
    python -m cartograph.data.vector_creator data/${directories[$i]} data/original_nav_vectors.txt
    python -m cartograph.data.popularity_creator data/${directories[$i]}
done
