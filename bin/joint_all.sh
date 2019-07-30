#!/bin/bash
#
# Joint algorithm which combines kmeans and labeling.
# Author: Lu Li, Jonathan Scott

set -e
set -x

listVar1="internet technology internet media science"
list2="0.5"
listVar1="0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.95"

for i in $list2
do
    # Step 0: Import the experiment utilities functions
    source ./bin/experiment-utils.sh

    # Assign variable name for label candidate we want (categories, links, keywork, etc)
    article_label_csv=article_hierarchical_categories.csv
    label_name_csv=hierarchical_category_names.csv

    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=0000 #$(get_experiment_id)

    # $(get_experiment_id)

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir internet ${exp_id})

    # Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
    # The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
    write_experiment_params ${exp_dir} weight 1
    # Step 4: If you needed to generate augmented vectors,
    # do so now from vanilla_vectors.csv in the experiment directory.

    # Step 5: Run algorithmic steps that are necessary.
    python -m cartograph.xy_embed.umap_embed \
            --map_directory ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --output_file /original_xy_embeddings.csv

    python -m kmeans+umap+labeling.joint_all \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --k 7 \
        --weight 0.01 \
        --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
        --article_keywords /Users/research/Documents/Projects/cartograph-alg/data/internet/$article_label_csv \
        --country_labels ${exp_dir}/country_labels.csv\
        --articles_to_labels data/internet/${article_label_csv} \
        --label_names data/internet/${label_name_csv} \
        --percentile 0.5 \
        --label_score tfidf \
        --cluster_groups /original_cluster_groups.csv \
        --output_file /country_labels.csv \
        --tf_idf_score_file /tf_idf_score.csv \
        --num_candidates 1  >>${exp_dir}/centroid.json

    python -m cartograph.xy_embed.umap_embed \
        --map_directory ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --output_file /new_xy_embeddings.csv \
        --cluster ${exp_dir}/new_cluster_groups.csv

    python -m cartograph.xy_embed.umap_embed \
        --map_directory ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --output_file /score_xy_embeddings.csv \
        --cluster ${exp_dir}/score_cluster_groups.csv


    options1="original new"
    for j in $options1
    do
        python -m cartograph.evaluation.cluster_validation_metrics --experiment ${exp_dir} \
                                                           --vectors ${exp_dir}/vanilla_vectors.csv \
                                                           --groups ${exp_dir}/${j}_cluster_groups.csv >>${exp_dir}/cluster_evaluation.json

    done
    options2="original new score"

    for j in $options2
    do

    python -m cartograph.evaluation.label_evaluator_against_study_gold_standards \
            --experiment_directory ${exp_dir} \
            --gold_standard  /Users/research/Documents/Projects/cartograph-alg/study/internet/kmeans_plain/gold_standard_labels.csv \
            --label_set ${exp_dir}/${j}_country_labels.csv \
            --k 7 >>${exp_dir}/experiment_params.json
    # Step 6: Generate JSON
    python -m cartograph.json_generator data/internet ${exp_dir}/ kk  ${j}_country_labels.csv ${j}_cluster_groups.csv ${j}_xy_embeddings.csv ${j}_domain.json

    # draw boundary
    python -m cartograph.border_creator ${exp_dir}/ ${j}_xy_embeddings.csv ${j}_cluster_groups.csv

    # Step 7: Run evaluation metrics and generate HTML & SVG
    python -m cartograph.svg_generator ${exp_dir}/ 1500 1500 hls ${j}_domain.json ${j}_graph.svg ${j}_country_labels.csv

    python -m cartograph.evaluation.modularity_evaluator --experiment ${exp_dir} \
                                                 --xy_embeddings_csv ${exp_dir}/${j}_xy_embeddings.csv \
                                                 --method nn \
                                                 --cluster_groups_csv ${exp_dir}/${j}_cluster_groups.csv >>${exp_dir}/mod_evaluation.json

    done


done