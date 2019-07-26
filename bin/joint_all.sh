#!/bin/bash
#
# Joint algorithm which combines kmeans and labeling.
# Author: Lu Li, Jonathan Scott

set -e
set -x

listVar1="food technology internet media science"
list2="food"
for i in $list2
do
    # Step 0: Import the experiment utilities functions
    source ./bin/experiment-utils.sh

    # Assign variable name for label candidate we want (categories, links, keywork, etc)
    article_label_csv=article_hierarchical_categories.csv
    label_name_csv=hierarchical_category_names.csv

    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=0028 #$(get_experiment_id)

    # $(get_experiment_id)

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir ${i} ${exp_id})

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
        --k 8 \
        --weight 0.9 \
        --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
        --article_keywords /Users/research/Documents/Projects/cartograph-alg/data/${i}/$article_label_csv \
        --country_names ${exp_dir}/country_labels.csv\
        --articles_to_labels data/${i}/${article_label_csv} \
        --label_names data/${i}/${label_name_csv} \
        --percentile 0.5 \
        --label_score tfidf \
        --cluster_groups /original_cluster_groups.csv \
        --output_file /country_labels.csv \
        --num_candidates 1

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


    options="original new score"
    for j in $options
    do

        # Step 6: Generate JSON
        python -m cartograph.json_generator data/${i} ${exp_dir}/ kk  ${j}_country_labels.csv ${j}_cluster_groups.csv ${j}_xy_embeddings.csv ${j}_domain.json

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