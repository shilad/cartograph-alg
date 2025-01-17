#!/bin/bash
#
# Author:

set -e
set -x

topic=georgraphy
cluster=8

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh


# Step 1: Initialize
exp_id=$(get_experiment_id)
exp_dir=$(prepare_experiment_dir $topic ${exp_id})
write_experiment_params ${exp_dir} weight 1

# Step 2: run UMAP
python -m beta.xy_embed.umap_embed \
        --map_directory ${exp_dir} \
        --vectors ${exp_dir}/vanilla_vectors.csv \
        --output_file /original_xy_embeddings.csv

# Step 3.a. run joint cluster algorithm
python -m beta.joint_cluster_builder \
    --experiment ${exp_dir} \
    --vectors ${exp_dir}/vanilla_vectors.csv \
    --xy_embeddings ${exp_dir}/original_xy_embeddings.csv\
    --articles_to_labels data/${topic}/article_hierarchical_categories.csv \
    --output_file ${exp_dir}/h_cat_cluster_groups.csv \
    --label_names data/$topic/hierarchical_category_names.csv \
    --k $cluster

#Step 3.c. get top labels, country, tfidf score, cluster algorithm, and label source.
python -m beta.feature_data_generator \
  --experiment ${exp_dir} \
  --articles_to_labels data/$topic/article_hierarchical_categories.csv \
  --label_names data/$topic/hierarchical_category_names.csv \
  --cluster_groups /h_cat_cluster_groups.csv \
  --output_file /h_cat_top_labels.csv \
  --label_source h_cat \
  --num_top_labels 500

# Step 3.b. run UMAP
python -m beta.xy_embed.umap_embed \
    --map_directory ${exp_dir} \
    --vectors ${exp_dir}/vanilla_vectors.csv \
    --output_file /new_xy_embeddings.csv \
    --cluster ${exp_dir}/h_cat_cluster_groups.csv
#

# Step 5: selects the label with top predicted borda as labels
python -m beta.borda_selector \
        --experiment ${exp_dir} \
        --borda_file h_cat_top_labels.csv

# Step 6: draw boundary
python -m beta.border_creator \
        ${exp_dir} /new_xy_embeddings.csv \
        /h_cat_cluster_groups.csv

# Step 7: Generate JSON, noise refers to using noise filtering algorithm (k means distance)
python -m beta.json_generator \
    --map_directory data/${topic} \
    --experiment ${exp_dir} \
    --filter_method kk  \
    --country_labels /final_labels.csv \
    --cluster_groups /h_cat_cluster_groups.csv \
    --embeddings /new_xy_embeddings.csv \
    --output_name /domain.json \
    --purpose experiment \
    --label_path NA


# Step 8: generate SVG

python -m beta.svg_generator \
    --map_directory ${exp_dir} \
    --width 1500 \
    --height 1500 \
    --color_palette hls \
    --json_file /domain.json \
    --output_file /graph.svg \
    --country_labels /final_labels.csv \
    --purpose experiment \
    --label_path NA