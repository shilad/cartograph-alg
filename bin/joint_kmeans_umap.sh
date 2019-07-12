

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
set -x

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh


# Step 1: Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
exp_id=0036

# $(get_experiment_id)

# Step 2: Prepare an experiment directory for a specific map.
exp_dir=$(prepare_experiment_dir food ${exp_id})

# Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
# The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
write_experiment_params ${exp_dir} spread 17 target_weight 0.5 clustering kmeans k 8 \
xy_embedding tsne

# Step 4: If you needed to generate augmented vectors,
# do so now from vanilla_vectors.csv in the experiment directory.

# Step 5: Run algorithmic steps that are necessary.

 python -m umap+kmeans.joint_umap_kmeans \
        --experiment ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --k 8


python -m cartograph.label_selector \
       --experiment ${exp_dir} \
        --articles_to_labels data/food/article_categories.csv \
        --label_names data/food/category_names.csv \
        --percentile 0.5 \
        --label_score tfidf


# Step 6: Generate JSON
python -m cartograph.json_generator data/food ${exp_dir} kk /original_cluster_groups.csv /original_xy_embeddings.csv /original_domain.json
python -m cartograph.json_generator data/food ${exp_dir} kk /cluster_groups.csv /original_xy_embeddings.csv /new_domain.json


# Step 7: Run evaluation metrics and generate HTML & SVG
python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted /original_domain.json /original_graph.svg
python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted /new_domain.json /new_graph.svg


# Step 8: Write evaluation metrics

python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /xy_embeddings.csv >>${exp_dir}/evaluation.json

# python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} /original_xy_embeddings.csv >>${exp_dir}/evaluation.json

python -m cartograph.evaluation.cluster_validation_metrics --experiment ${exp_dir} \
                                                           --vectors ${exp_dir}/vanilla_vectors.csv \
                                                           --groups ${exp_dir}/cluster_groups.csv >>${exp_dir}/evaluation.json

python -m cartograph.html_generator ${exp_dir}