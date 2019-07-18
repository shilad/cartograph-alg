#!/usr/bin/env bash
# Generates the CSV necessary to run the clustering and labeling Mechanical Turk Study
#
# source ~/venv/cartograph-alg/bin/activate
#
# Author: Lily Irvin


set -e
set -x


# Assign wikiproject for data directory
projects="food internet media technology"
project=(food internet media technology)

exp_ids=(kmeans_augmented kmeans_plain LDA)
thresholds=(150000 50000 20000 120000)


for i in {0..3}
do
    echo ${project[$i]}

    for x in {0..2}
    do
        python -m study.data.top_cluster_labels study/${project[$i]}/${exp_ids[$x]} ${exp_ids[$x]}
        python -m study.data.get_final_articles study/${project[$i]}/${exp_ids[$x]} data/${project[$i]} ${exp_ids[$x]} ${thresholds[$i]}
    done

    python -m study.data.get_project_groups study/${project[$i]} ${project[$i]}
done

python -m study.data.get_mturk_csv study ${projects}
