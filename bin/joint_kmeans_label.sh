

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

# Assign variable name for label candidate we want (categories, links, keywork, etc)
article_label_csv=article_keywords.csv
label_name_csv=keyword_names.csv

# Step 1: Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
exp_id=0004 #$(get_experiment_id)

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
        --vectors ${exp_dir}/vanilla_vectors.csv

python -m cartograph.label_selector \
        --experiment ${exp_dir} \
        --articles_to_labels data/food/${article_label_csv} \
        --label_names data/food/${label_name_csv} \
        --percentile 0.3 \
        --label_score tfidf \
        --cluster_groups /original_cluster_groups.csv \
        --output_file /country_labels.csv


python -m kmeans+label.joint_label_kmeans \
    --experiment ${exp_dir} \
    --vectors ${exp_dir}/vanilla_vectors.csv \
    --k 10 \
    --weight 5 \
    --article_keywords /Users/research/Documents/Projects/cartograph-alg/data/food/article_keywords.csv \
    --country_names ${exp_dir}/country_labels.csv


python -m cartograph.label_selector \
        --experiment ${exp_dir} \
        --articles_to_labels data/food/${article_label_csv} \
        --label_names data/food/${label_name_csv} \
        --percentile 0.3 \
        --label_score tfidf \
        --cluster_groups /cluster_groups.csv \
        --output_file /new_country_labels.csv


# Step 6: Generate JSON
python -m cartograph.json_generator data/food ${exp_dir} kk  /country_labels.csv /original_cluster_groups.csv /xy_embeddings.csv /original_domain.json
python -m cartograph.json_generator data/food ${exp_dir} kk /new_country_labels.csv /cluster_groups.csv /xy_embeddings.csv /new_domain.json



# Step 7: Run evaluation metrics and generate HTML & SVG
python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted /original_domain.json /original_graph.svg /country_labels.csv
python -m cartograph.svg_generator ${exp_dir} 1500 1500 muted /new_domain.json /new_graph.svg /new_country_labels.csv

