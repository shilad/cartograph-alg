#!/bin/bash
#
# Joint algorithm which combines kmeans and labeling.
# Author: Lu Li, Jonathan Scott

set -e
set -x

listVar1="technology internet media science"
for i in $listVar1
do
    # Step 0: Import the experiment utilities functions
    source ./bin/experiment-utils.sh

    # Assign variable name for label candidate we want (categories, links, keywork, etc)
    article_label_csv=article_keywords.csv
    label_name_csv=keyword_names.csv

    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=0007 #$(get_experiment_id)

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
            --vectors ${exp_dir}/vanilla_vectors.csv



    python -m kmeans+label.joint_label_kmeans \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --k 8 \
        --weight 0.5 \
        --article_keywords /Users/research/Documents/Projects/cartograph-alg/data/${i}/article_keywords.csv \
        --country_names ${exp_dir}/country_labels.csv\
        --articles_to_labels data/${i}/${article_label_csv} \
        --label_names data/${i}/${label_name_csv} \
        --percentile 0.5 \
        --label_score tfidf \
        --cluster_groups /original_cluster_groups.csv \
        --output_file /country_labels.csv \
        --num_candidates 4
    options="original new score"
    for j in $options
    do

        # Step 6: Generate JSON
        python -m cartograph.json_generator data/${i} ${exp_dir}/ kk  ${j}_country_labels.csv ${j}_cluster_groups.csv xy_embeddings.csv ${j}_domain.json

        # draw boundary
        python -m cartograph.border_creator ${exp_dir}/ xy_embeddings.csv ${j}_cluster_groups.csv

        # Step 7: Run evaluation metrics and generate HTML & SVG
        python -m cartograph.svg_generator ${exp_dir}/ 1500 1500 muted ${j}_domain.json ${j}_graph.svg ${j}_country_labels.csv

    done
done