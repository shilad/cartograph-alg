#!/bin/bash
#
# Generates maps for the user study using LDA.
#
# source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/augmentation-experiment.sh
#
# Author: Lily Irvin


set -e
set -x


# Assign wikiproject for data directory
projects=(food internet media technology)


# Assign variable name for label candidate we want (categories, links, keyword, etc)
label_type=(h_cat links key_phrases key_words)
article_label_csv=(article_hierarchical_categories.csv article_links.csv article_keyphrases.csv article_keywords.csv)
label_name_csv=(hierarchical_category_names.csv link_names.csv keyphrases_names.csv keyword_names.csv)
label_score=tfidf


# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh


for i in {0..3}
do
    echo ${projects[$i]}
    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=LDA


    # Prepares an experiment directory
    # Returns the name of the experiment directory for the map
    #
    prepare_experiment_dir () {
        map_name=${projects[$i]}
        exp_id=$1

        exp_dir=study/${map_name}/${exp_id}

        echo "using directory $exp_dir" >&2
        mkdir -p ${exp_dir}

        # Copy in the script used to run the experiment
        cp -p $0 ${exp_dir}

        echo ${exp_dir}
    }


    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir ${exp_id})


#    # Step 3: Run LDA:
#    python -m cartograph.topic_finder ${exp_dir} data/${projects[$i]} 10 /LDA_labels.csv /article_topic_distribution.csv
#    python -m cartograph.topic_finder ${exp_dir} data/${projects[$i]} 50 /LDA_labels_50.csv /article_topic_distribution_50.csv
#
#
#    # Step 4: create cluster csv for label_selector:
#    python -m study.data.LDA_cluster_builder ${exp_dir}


    for x in {0..3}
    do
        label_path=${exp_dir}/labels/${label_type[$x]}

        # Step 4: Run other labeling schemes:
        python -m cartograph.label_selector \
            --experiment ${exp_dir} \
            --articles_to_labels data/${projects[$i]}/${article_label_csv[$x]} \
            --label_names data/${projects[$i]}/${label_name_csv[$x]} \
            --label_score ${label_score} \
            --percentile 1 \
            --purpose study \
            --label_path ${label_path} \
            --cluster_groups /cluster_groups.csv \
            --output_file /final_labels.csv \
            --soft_labeling false \
            --num_candidates 0
    done


#    # Step 5: Create lda_labels
#    python -m cartograph.data.label_creator_lda data/${projects[$i]} ${exp_dir}/article_topic_distribution.csv ${exp_dir}/labels/LDA_labels/LDA_labels_50.csv


done
#!/usr/bin/env bash