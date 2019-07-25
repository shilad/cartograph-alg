#!/usr/bin/env bash

#exp_dir=$EXP_DIR
#topic=$TOPIC
#article_label_csv=$ARTICLE_LABEL_CSV
#label_name_csv=$LABEL_NAME_CSV
topic=food
article_label_csv=article_keywords.csv
label_name_csv=keyword_names.csv

# Step 0: Import the experiment utilities functions
source ./bin/experiment-utils.sh

# Step 1: Get the experiment id. This is *not* map specific.
# An experiment id can be used for multiple maps.
exp_id=$(get_experiment_id)

# Step 2: Prepare an experiment directory for a specific map.
exp_dir=$(prepare_experiment_dir ${topic} ${exp_id})

# Step 3: You MUST pass any configuration parameters important to the experiment as key-value pairs.
# The example below passes the equivalent of { "spread" : "17", "target_weight" : "0.5" }.
write_experiment_params ${exp_dir} num_clusters 8 labels keywords xy_embedding tsne cluster_method kmeans


python -m cartograph.vector_augmenter \
    --experiment ${exp_dir} \
    --vectors vanilla_vectors.csv \
    --label_vectors data/${topic}/${article_label_csv} \
    --method label

python -m cartograph.cluster_builder \
    --experiment ${exp_dir} \
    --vectors label_augmented_vectors.csv \
    --clustering kmeans \
    --k 8

python -m cartograph.label_selector \
    --experiment ${exp_dir} \
    --articles_to_labels data/${topic}/${article_label_csv} \
    --label_names data/${topic}/${label_name_csv} \
    --label_score tfidf \
    --percentile 0.3

# augment cluster for embedding
# could also use method all, but it doesn't make a big difference
python -m cartograph.vector_augmenter \
    --experiment ${exp_dir} \
    --vectors vanilla_vectors.csv \
    --method cluster

python -m cartograph.xy_embed.tsne_embed \
    --experiment ${exp_dir} \
    --vectors cluster_augmented_vectors.csv

python -m cartograph.border_creator \
    --experiment ${exp_dir}

python -m cartograph.json_generator \
    --map_directory data/${topic} \
    --experiment ${exp_dir} \
    --filter_method noise

python -m cartograph.svg_generator \
    --experiment ${exp_dir}  \
    --width 1500 \
    --height 1500 \
    --color_palette hls

python -m cartograph.evaluation.xy_embedding_validation ${exp_dir} >>${exp_dir}/evaluation.json

python -m cartograph.evaluation.modularity_evaluator \
    --experiment ${exp_dir} \
    --method nn >>${exp_dir}/evaluation.json

#python -m cartograph.evaluation.cluster_validation_metrics \
#    --experiment ${exp_dir} \
#    --vectors ${exp_dir}/label_augmented_vectors.csv \
#    --groups ${exp_dir}/cluster_groups.csv >> ${exp_dir}/evaluation.json


# Step 8: Output the html for visualization
python -m cartograph.html_generator \
    --experiment ${exp_dir}