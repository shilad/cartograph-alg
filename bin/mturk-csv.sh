#!/usr/bin/env bash
# Generates the CSV necessary to run the clustering and labeling Mechanical Turk Study
#
# source ~/venv/cartograph-alg/bin/activate
#
# Author: Lily Irvin


set -e
set -x


# Assign wikiproject for data directory
#projects=(food internet media technology)
projects=(food internet media technology)

exp_ids=(kmeans_augmented kmeans_plain LDA)
treatments=(augmented plain LDA)
thresholds=(120000 50000 10000 120000)


for i in {1..3}
do
    echo ${projects[$i]}

    for x in {0..2}
    do
#        python -m study.data.top_cluster_labels study/${projects[$i]}/${exp_ids[$x]} ${exp_ids[$x]}
        python -m study.data.get_final_articles study/${projects[$i]}/${exp_ids[$x]} data/${projects[$i]} ${treatments[$x]} ${thresholds[$i]}
    done

done