#!/bin/bash
#
# Creates maps for all of the wikiprojects used in the Mechanical Turk user study
#
# source ~/venv/cartograph-alg/bin/activate
#

set -e
set -x

projects=(Geography Football)
directories=(georgraphy football)
#num_of_articles=(93811 353772)


for i in {0..0}
do
    echo ${projects[$i]}
#    python -m beta.data.domain_concept_creator data/${directories[$i]} ${projects[$i]} ${num_of_articles[$i]}
    python -m beta.data.label_creator_hierarchical_cat data/${directories[$i]}
#    python -m beta.data.label_creator_links data/${directories[$i]}
#    python -m beta.data.text_getter data/${directories[$i]}
#    python -m beta.data.label_creator_keywords data/${directories[$i]}
#    python -m beta.data.label_creator_key_phrases data/${directories[$i]}
#    python -m beta.data.label_creator_combined data/${directories[$i]}
#    python -m beta.data.vector_creator data/${directories[$i]} data/original_nav_vectors.txt
#    python -m beta.data.popularity_creator data/${directories[$i]}
done

#

#for i in {0..0}
#do
#python -m beta.topic_finder data/georgraphy data/georgraphy \
#                                 15 /LDA_labels.csv /article_topic_distribution.csv
#
#python -m beta.data.label_creator_lda data/georgraphy data/georgraphy/article_topic_distribution.csv \
#                                        data/georgraphy/labels/LDA_labels/LDA_labels.csv
#done
#
#for i in {1..1}
#do
#    echo ${projects[$i]}
##    python -m beta.data.domain_concept_creator data/${directories[$i]} ${projects[$i]} ${num_of_articles[$i]}
##    python -m beta.data.label_creator_hierarchical_cat data/${directories[$i]}
#    python -m beta.data.label_creator_links data/${directories[$i]}
#    python -m beta.data.text_getter data/${directories[$i]}
#    python -m beta.data.label_creator_keywords data/${directories[$i]}
#    python -m beta.data.label_creator_key_phrases data/${directories[$i]}
#    python -m beta.data.label_creator_combined data/${directories[$i]}
##    python -m beta.data.vector_creator data/${directories[$i]} data/original_nav_vectors.txt.bz2
#    python -m beta.data.popularity_creator data/${directories[$i]}
#done
