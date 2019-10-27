#!/bin/bash
#
# Example shell script to run an experiment
# Before running this, you must activate Python.
#
# For example: source ~/venv/cartograph-alg/bin/activate
#
# Run this script using ./bin/example-experiment.sh
#
# Author: Shilad Sen, Jonathan Scott

set -e
set -x

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh


# Step 1: Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
# specifiy an experiment
exp_id=0007 #$(get_experiment_id)

# Step 2: Prepare an experiment directory for a specific map.
exp_dir=$(prepare_experiment_dir food ${exp_id})
gold_standard=study/food/kmeans_augmented/gold_standard_labels.csv

# Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
# The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
#write_experiment_params ${exp_dir} spread 17 target_weight 0.5 clustering kmeans k 8 \
#xy_embedding tsne

# Step 4: If you needed to generate augmented vectors,
# do so now from vanilla_vectors.csv in the experiment directory.

# Step 5: Run algorithmic steps that are necessary.

python -m cartograph.create_label_training-data \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --cluster_groups ${exp_dir}/original_cluster_groups.csv \
        --gold_standard ${gold_standard} \
        --output_file ${exp_dir}/labeling_training_data.csv



