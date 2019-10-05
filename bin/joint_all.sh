#!/bin/bash
#
# Joint algorithm which combines kmeans and labeling.
# Author: Lu Li, Jonathan Scott

set -e
set -x

listVar1="food"
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
    exp_id=$(get_experiment_id)

    # $(get_experiment_id)

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir food ${exp_id})

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
        --weight 0.5 \
        --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
        --article_keywords data/food/$article_label_csv \
        --country_labels ${exp_dir}/country_labels.csv \
        --articles_to_labels data/food/${article_label_csv} \
        --label_names data/food/${label_name_csv} \
        --percentile 0.5 \
        --label_score tfidf \
        --cluster_groups /original_cluster_groups.csv \
        --output_file /country_labels.csv \
        --tf_idf_score_file /tf_idf_score.csv \
        --num_candidates 4  >>${exp_dir}/centroid.json

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
    options2="original new"

    for j in $options2
    do

#    python -m cartograph.evaluation.label_evaluator_against_study_gold_standards \
#            --experiment_directory ${exp_dir} \
#            --gold_standard  study/food/kmeans_plain/gold_standard_labels.csv \
#            --label_set ${exp_dir}/${j}_country_labels.csv \
#            --k 7 >>${exp_dir}/label_evaluation.json

    # Step 6: Generate JSON
    python -m cartograph.json_generator \
    --map_directory data/food \
    --experiment ${exp_dir}/ \
    --filter_method kk  \
    --country_labels ${j}_country_labels.csv \
    --cluster_groups ${j}_cluster_groups.csv \
    --embeddings ${j}_xy_embeddings.csv \
    --output_name ${j}_domain.json \
    --purpose experiment \
    --label_path na

    # draw boundary
    python -m cartograph.border_creator ${exp_dir}/ ${j}_xy_embeddings.csv ${j}_cluster_groups.csv

    # Step 7: Run evaluation metrics and generate HTML & SVG
    python -m cartograph.svg_generator \
    --map_directory ${exp_dir}/ \
    --width 1500 \
    --height 1500 \
    --color_palette muted \
    --json_file ${j}_domain.json \
    --output_file ${j}_graph.svg \
    --country_labels ${j}_country_labels.csv \
    --purpose experiment \
    --label_path na

    python -m cartograph.evaluation.modularity_evaluator --experiment ${exp_dir} \
                                                 --xy_embeddings_csv ${exp_dir}/${j}_xy_embeddings.csv \
                                                 --method nn \
                                                 --cluster_groups_csv ${exp_dir}/${j}_cluster_groups.csv >>${exp_dir}/mod_evaluation.json

    done


done