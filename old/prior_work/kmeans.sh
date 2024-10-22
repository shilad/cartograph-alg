#!/bin/bash
#
# Example shell script to run an experiment
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/example-experiment.sh
#
# Author: Shilad Sen

set -e

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh
# $(seq 8 0 8)

for i in 8
    do

    # Step 1: Get the experiment id. This is *not* map specific.
    # An experiment id can be used for multiple maps.
    exp_id=0021

    # Step 2: Prepare an experiment directory for a specific map.
    exp_dir=$(prepare_experiment_dir food ${exp_id})

    # Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
    # The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
    write_experiment_params ${exp_dir} num_clusters $i


    # Step 5: Run algorithmic steps that are necessary.

     python -m cartograph.cluster_builder \
            --experiment ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --clustering kmeans \
            --k $i # \
           # --min_size 2


     python -m cartograph.label_selector \
            --experiment ${exp_dir} \
            --articles_to_labels data/food/article_categories.csv \
            --label_names data/food/category_names.csv

    # Step 4: If you needed to generate augmented vectors,
    # do so now from vanilla_vectors.csv in the experiment directory.

    python -m cartograph.vector_augmenter \
            --experiment ${exp_dir} \
            --vectors ${exp_dir}/vanilla_vectors.csv \
            --label_vectors ${exp_dir}/chocolate_vectors.csv \
            --method cluster \
            --cluster_vectors ${exp_dir}/cluster_groups.csv
    python -m cartograph.xy_embed.tsne_embed \
            --experiment ${exp_dir} \
           --vectors ${exp_dir}/augmented_vectors.csv

    # Step 6: Generate JSON
    python -m cartograph.json_generator data/food ${exp_dir}



    python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted

    # Step 7: Run evaluation metrics and generate HTML & SVG

    python -m cartograph.evaluation.modularity_evaluator \
            --experiment ${exp_dir} \
            --xy_embeddings_csv ${exp_dir}/xy_embeddings.csv \
            --method nn \
            --cluster_groups_csv ${exp_dir}/cluster_groups.csv >> ${exp_dir}/params.json
    python -m cartograph.evaluation.cluster_validation_metrics \
            --experiment ${exp_dir} \
            --groups ${exp_dir}/cluster_groups.csv \
            --vectors ${exp_dir}/vanilla_vectors.csv >> ${exp_dir}/params.json

    done
