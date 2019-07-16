#!/usr/bin/env bash
# Generates the CSV necessary to run the clustering and labeling Mechanical Turk Study
#
# source ~/venv/cartograph-alg/bin/activate
#
# Author: Lily Irvin


set -e
set -x


# Assign wikiproject for data directory
projects=(food internet media technology)

exp_ids=(kmeans_augmented kmeans_plain LDA)


for i in {0..3}
do
    echo ${projects[$i]}

    for x in {0..2}
    do
        python -m study.data.top_cluster_labels study/${projects[$i]}/${exp_ids[$x]} ${exp_ids[$x]}
    done

done